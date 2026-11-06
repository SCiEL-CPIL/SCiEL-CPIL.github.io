import subprocess, os
r = subprocess.run(['bash','-lc','find / -type f -name "science_tcs_expansion.py" 2>/dev/null | head -5'], capture_output=True, text=True, timeout=60)
paths=[l for l in r.stdout.strip().splitlines() if l]
print("PATHS", paths)
for p in paths:
    print("=====", p)
    print(open(p).read())
