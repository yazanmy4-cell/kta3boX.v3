#!/usr/bin/env python3
"""
YT2Text - paste a YouTube link, pick how many clips and how long,
get MP3 clips + transcript for each. Free and local.
Run:  python server.py   then open http://localhost:8000
"""
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
COOKIES = BASE / "cookies.txt"
PORT = int(os.environ.get("PORT", "8000"))
LIBRARY = BASE / "library.json"

JOBS = {}
LIB = {"items": []}
LOCK = threading.Lock()
_MODELS = {}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MIN_SPEECH_RATIO = 0.15   # 15% of clip must contain speech to be "nice"


# ------------------------------------------------------------------ library
def load_library():
    global LIB
    if LIBRARY.exists():
        try:
            LIB = json.loads(LIBRARY.read_text(encoding="utf-8"))
            LIB.setdefault("items", [])
        except Exception:
            LIB = {"items": []}


def save_library():
    tmp = LIBRARY.with_suffix(".tmp")
    tmp.write_text(json.dumps(LIB, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, LIBRARY)


def library_add(item):
    with LOCK:
        LIB["items"].insert(0, item)   # newest first
        LIB["items"] = LIB["items"][:200]
        save_library()


def library_remove(item_id):
    with LOCK:
        LIB["items"] = [x for x in LIB["items"] if x.get("id") != item_id]
        save_library()


# ------------------------------------------------------------------ helpers
def parse_float(v, default=None):
    v = (v or "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        raise ValueError("bad number")


def stamp(t, sep):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def short_stamp(t):
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_transcript(segs, fmt, stamps):
    if fmt == "srt":
        out = []
        for i, s in enumerate(segs, 1):
            out.append(f"{i}\n{stamp(s['start'], ',')} --> {stamp(s['end'], ',')}\n{s['text']}\n")
        return "\n".join(out)
    if fmt == "vtt":
        out = ["WEBVTT\n"]
        for s in segs:
            out.append(f"{stamp(s['start'], '.')} --> {stamp(s['end'], '.')}\n{s['text']}\n")
        return "\n".join(out)
    lines = []
    for s in segs:
        lines.append(f"[{short_stamp(s['start'])}] {s['text']}" if stamps else s["text"])
    return "\n".join(lines) + "\n"


def safe_name(title):
    n = re.sub(r"[^\w\- ]+", "", title or "", flags=re.U).strip()
    return (n[:50] or "clip").strip()


def fmt_dur(sec):
    sec = int(round(sec))
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def get_model(name):
    from faster_whisper import WhisperModel
    with LOCK:
        if name not in _MODELS:
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            compute = os.environ.get("WHISPER_COMPUTE", "int8")
            _MODELS[name] = WhisperModel(name, device=device, compute_type=compute)
        return _MODELS[name]


def cut_mp3(src: Path, dst: Path, start: float, end: float, kbps: int):
    """Cut [start, end] from src into dst using ffmpeg. Re-encode at kbps."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(src),
        "-codec:a", "libmp3lame", "-b:a", f"{kbps}k",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {r.stderr.strip()[:300]}")


def ffprobe_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=20)
        return float(r.stdout.strip() or 0)
    except Exception:
        return 0.0


# ------------------------------------------------------------------ job
def run_job(jid, p):
    job = JOBS[jid]
    d = OUT / jid

    def log(m):
        job["msgs"] = (job["msgs"] + [ANSI.sub("", str(m))])[-8:]

    def stage(key, label, prog):
        job.update(stage=key, stage_label=label, progress=prog)

    try:
        import yt_dlp
        from yt_dlp.utils import download_range_func

        d.mkdir(exist_ok=True)
        base_opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
        if COOKIES.exists():
            base_opts["cookiefile"] = str(COOKIES)

        # 1) info
        stage("info", "Reading video info", 4)
        log("Contacting YouTube...")
        with yt_dlp.YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(p["url"], download=False)
        title = info.get("title") or "video"
        duration = info.get("duration") or 0
        uploader = info.get("uploader") or ""
        job.update(video_title=title, uploader=uploader, video_duration=duration)
        log(f"Found: {title} ({fmt_dur(duration)})")

        # 2) download full audio -> mp3
        stage("audio", "Downloading audio", 8)
        opts = dict(base_opts)
        opts.update(
            format="bestaudio/best",
            outtmpl=str(d / "source.%(ext)s"),
            postprocessors=[
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                 "preferredquality": str(p["kbps"])}
            ],
        )

        def hook(h):
            if h.get("status") == "downloading":
                tot = h.get("total_bytes") or h.get("total_bytes_estimate") or 0
                if tot:
                    job["progress"] = 8 + 22 * (h.get("downloaded_bytes", 0) / tot)
            elif h.get("status") == "finished":
                log("Converting to MP3...")

        opts["progress_hooks"] = [hook]
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([p["url"]])

        src_mp3 = d / "source.mp3"
        if not src_mp3.exists():
            raise RuntimeError("MP3 was not created. Is ffmpeg installed on PATH?")
        real_dur = ffprobe_duration(src_mp3) or duration or 0
        log(f"Audio ready ({real_dur:.1f}s)")

        # 3) plan clips
        stage("plan", "Planning clips", 30)
        clip_n = p["clip_count"]
        clip_len = p["clip_duration"]
        total_needed = clip_n * clip_len
        if total_needed > real_dur:
            clip_n = max(1, int(real_dur // clip_len))
            log(f"Video shorter than requested — reducing to {clip_n} clips")

        # spread across video evenly
        step = (real_dur - clip_len) / max(1, clip_n - 1) if clip_n > 1 else 0
        planned = []
        for i in range(clip_n):
            s = i * step
            e = min(real_dur, s + clip_len)
            planned.append({"index": i, "start": s, "end": e})

        # 4) transcribe FULL audio once (faster than per-clip)
        full_segs = []
        language = ""
        if p["transcribe"]:
            stage("transcribe", "Loading speech model", 34)
            log(f"Loading Whisper model: {p['model']}")
            model = get_model(p["model"])
            stage("transcribe", "Finding speech", 38)
            segments, tinfo = model.transcribe(
                str(src_mp3),
                language=None if p["lang"] == "auto" else p["lang"],
                vad_filter=True, beam_size=5,
            )
            language = tinfo.language or ""
            log(f"Language: {language}")
            total_t = tinfo.duration or real_dur or 1
            for seg in segments:
                t = seg.text.strip()
                if not t:
                    continue
                full_segs.append({"start": seg.start, "end": seg.end, "text": t})
                job["progress"] = 38 + 20 * min(1.0, seg.end / total_t)
                job["stage_label"] = f"Finding speech - {short_stamp(seg.end)} / {short_stamp(total_t)}"
            log(f"Found {len(full_segs)} speech segments")

        # 5) decide which clips are "nice"
        stage("filter", "Choosing best clips", 60)

        def speech_ratio(c):
            tot = c["end"] - c["start"]
            if tot <= 0:
                return 0.0
            covered = 0.0
            for s in full_segs:
                a = max(s["start"], c["start"])
                b = min(s["end"], c["end"])
                if b > a:
                    covered += (b - a)
            return covered / tot

        for c in planned:
            c["speech_ratio"] = speech_ratio(c)

        # if transcribe is off, keep all (can't judge)
        if p["transcribe"]:
            good = [c for c in planned if c["speech_ratio"] >= MIN_SPEECH_RATIO]
            if not good:
                # fallback: keep the top half by speech ratio
                planned.sort(key=lambda x: x["speech_ratio"], reverse=True)
                good = planned[:max(1, len(planned) // 2)]
            planned = sorted(good, key=lambda x: x["start"])

        if not planned:
            raise RuntimeError("No usable clips found in this video.")

        log(f"Keeping {len(planned)} of {clip_n} clips")

        # 6) cut + save each clip
        stage("cut", "Cutting clips", 65)
        results = []
        base_name = safe_name(title)

        for i, c in enumerate(planned):
            dur = c["end"] - c["start"]
            clip_name = f"{base_name}_clip{i+1:02d}.mp3"
            clip_path = d / clip_name
            try:
                cut_mp3(src_mp3, clip_path, c["start"], c["end"], p["kbps"])
            except Exception as e:
                log(f"clip {i+1} cut failed: {e}")
                continue

            # build clip transcript from segments within [start, end]
            clip_text = ""
            clip_tfile = None
            if p["transcribe"]:
                rel_segs = []
                for s in full_segs:
                    a = max(s["start"], c["start"])
                    b = min(s["end"], c["end"])
                    if b > a:
                        rel_segs.append({
                            "start": a - c["start"],
                            "end": b - c["start"],
                            "text": s["text"],
                        })
                clip_text = build_transcript(rel_segs, p["format"], p["stamps"])
                clip_tfile = d / f"{base_name}_clip{i+1:02d}.{p['format']}"
                clip_tfile.write_text(clip_text, encoding="utf-8")

            item = {
                "index": i + 1,
                "filename": clip_name,
                "start": c["start"],
                "end": c["end"],
                "duration": dur,
                "speech_ratio": round(c["speech_ratio"], 3),
                "has_text": bool(clip_text),
                "text_filename": clip_tfile.name if clip_tfile else None,
                "audio_url": f"/api/file/{jid}/{quote(clip_name)}",
                "audio_download": f"/api/file/{jid}/{quote(clip_name)}?dl=1",
                "text": clip_text,
            }
            if clip_tfile:
                item["text_url"] = f"/api/file/{jid}/{quote(clip_tfile.name)}"
                item["text_download"] = f"/api/file/{jid}/{quote(clip_tfile.name)}?dl=1"

            results.append(item)
            job["progress"] = 65 + 30 * ((i + 1) / len(planned))
            job["stage_label"] = f"Cutting clips - {i+1}/{len(planned)}"

        if not results:
            raise RuntimeError("All cuts failed.")

        # 7) zip + library entry
        stage("save", "Packing", 96)
        zpath = d / "all.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for it in results:
                zp = d / it["filename"]
                if zp.exists():
                    z.write(zp, it["filename"])
                if it.get("text_filename"):
                    tp = d / it["text_filename"]
                    if tp.exists():
                        z.write(tp, it["text_filename"])

        thumb = ""
        if info.get("thumbnail"):
            thumb = info.get("thumbnail")

        job.update(
            clips=results,
            zip_url=f"/api/zip/{jid}",
            format=p["format"],
            language=language,
            transcribed=bool(p["transcribe"]),
            stage="done", progress=100, stage_label="Done",
        )

        library_add({
            "id": jid,
            "title": title,
            "uploader": uploader,
            "thumbnail": thumb,
            "duration": real_dur,
            "clip_count": len(results),
            "clip_duration": p["clip_duration"],
            "format": p["format"],
            "language": language,
            "created_at": time.time(),
            "zip_url": f"/api/zip/{jid}",
            "clips": results,
        })
    except Exception as e:  # noqa: BLE001
        msg = ANSI.sub("", str(e)).strip() or e.__class__.__name__
        if "ffmpeg" in msg.lower() and "not found" in msg.lower():
            msg += " Install ffmpeg and make sure it is on PATH."
        if isinstance(e, ModuleNotFoundError):
            msg = f"Missing Python package: {e.name}. Run: pip install -r requirements.txt"
        job.update(stage="error", error=msg)


STEPS_FULL = [
    {"key": "info", "label": "Reading info"},
    {"key": "audio", "label": "Downloading audio"},
    {"key": "transcribe", "label": "Finding speech"},
    {"key": "filter", "label": "Choosing best clips"},
    {"key": "cut", "label": "Cutting clips"},
    {"key": "save", "label": "Packing"},
]


def create_job(body):
    url = (body.get("url") or "").strip()
    if not re.match(r"^https?://", url, re.I):
        raise ValueError("Paste a valid video link.")

    try:
        clip_count = int(body.get("clip_count", 5))
        clip_duration = float(body.get("clip_duration", 15))
    except (TypeError, ValueError):
        raise ValueError("Clip count and duration must be numbers.")
    clip_count = max(1, min(50, clip_count))
    clip_duration = max(3.0, min(120.0, clip_duration))

    fmt = body.get("format", "txt")
    if fmt not in ("txt", "srt", "vtt"):
        fmt = "txt"
    model = body.get("model", "small")
    if model not in ("tiny", "base", "small", "medium"):
        model = "small"
    kbps = int(body.get("kbps", 192))
    if kbps not in (96, 128, 192, 256, 320):
        kbps = 192
    lang = re.sub(r"[^A-Za-z\-]", "", str(body.get("lang", "auto")))[:8] or "auto"

    p = dict(
        url=url,
        clip_count=clip_count,
        clip_duration=clip_duration,
        format=fmt,
        model=model,
        kbps=kbps,
        lang=lang,
        stamps=bool(body.get("stamps", False)),
        transcribe=bool(body.get("transcribe", True)),
    )
    jid = uuid.uuid4().hex[:12]
    steps = STEPS_FULL if p["transcribe"] else [s for s in STEPS_FULL if s["key"] not in ("transcribe", "filter")]
    JOBS[jid] = dict(id=jid, stage="queued", stage_label="Starting...", progress=1, msgs=[], steps=steps)
    threading.Thread(target=run_job, args=(jid, p), daemon=True).start()
    return jid


# ------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    server_version = "YT2Text/2.0"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, download=False):
        if not path.is_file():
            return self._json({"error": "Not found"}, 404)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() in (".txt", ".srt", ".vtt"):
            ctype = "text/vtt; charset=utf-8" if path.suffix.lower() == ".vtt" else "text/plain; charset=utf-8"
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name)}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        if path in ("/", "/index.html"):
            return self._file(BASE / "index.html")

        # --- API ---
        m = re.match(r"^/api/status/([0-9a-f]{12})$", path)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self._json({"error": "Unknown job"}, 404)
            return self._json({k: v for k, v in job.items() if not k.startswith("_")})

        if path == "/api/library":
            load_library()
            # strip big text fields from listing
            items = []
            for it in LIB["items"]:
                light = {k: v for k, v in it.items() if k != "clips"}
                light["clip_count"] = len(it.get("clips", []))
                items.append(light)
            return self._json({"items": items})

        m = re.match(r"^/api/library/([0-9a-f]{12})$", path)
        if m:
            load_library()
            for it in LIB["items"]:
                if it.get("id") == m.group(1):
                    return self._json(it)
            return self._json({"error": "Not found"}, 404)

        m = re.match(r"^/api/file/([0-9a-f]{12})/(.+)$", path)
        if m:
            name = Path(unquote(m.group(2))).name
            return self._file(OUT / m.group(1) / name, download="dl=1" in (u.query or ""))

        m = re.match(r"^/api/zip/([0-9a-f]{12})$", path)
        if m:
            return self._file(OUT / m.group(1) / "all.zip", download=True)

        self._json({"error": "Not found"}, 404)

    def do_DELETE(self):
        u = urlparse(self.path)
        m = re.match(r"^/api/library/([0-9a-f]{12})$", u.path)
        if not m:
            return self._json({"error": "Not found"}, 404)
        item_id = m.group(1)
        library_remove(item_id)
        # remove files
        target = OUT / item_id
        if target.exists():
            try:
                shutil.rmtree(target)
            except Exception:
                pass
        return self._json({"ok": True})

    def do_POST(self):
        if urlparse(self.path).path != "/api/create":
            return self._json({"error": "Not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            jid = create_job(body)
            self._json({"id": jid})
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 500)


def main():
    load_library()
    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg not found on PATH. MP3 conversion will fail until you install it.")
    if not shutil.which("ffprobe"):
        print("WARNING: ffprobe not found on PATH. Duration detection may fail.")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"YT2Text running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
