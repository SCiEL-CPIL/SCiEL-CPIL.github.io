import time

_P = 1009
_M = 5


def _precompute():
    roots = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in roots:
            roots[r] = x
    return roots


def construct():
    start_time = time.monotonic()
    deadline = start_time + 55.0

    roots = _precompute()
    root_set = set(roots.keys())

    max_s = _P - _M + 1  # 1005, so s in 1..1004

    for s in range(1, max_s):
        if time.monotonic() > deadline:
            raise RuntimeError("Deadline exceeded before finding a run")

        # Check if s, s+1, s+2, s+3, s+4 are all in root_set
        if (s in root_set and
                (s + 1) in root_set and
                (s + 2) in root_set and
                (s + 3) in root_set and
                (s + 4) in root_set):
            return {'start': s, 'roots': [roots[s], roots[s + 1], roots[s + 2], roots[s + 3], roots[s + 4]]}

    raise RuntimeError("No run found")
