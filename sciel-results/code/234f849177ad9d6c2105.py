"""Color [1,128] so every nonconstant 4-AP uses >=3 distinct colors.
Optimal is 2 colors iff the 4-AP hypergraph on [1,128] is 2-colorable
(no monochromatic 4-AP). We search for such a 2-coloring; if found it is
the minimum possible (1 color is impossible)."""
import time
import random

_N = 128
_DEADLINE = 75.0

def _build_aps():
    aps = []
    for d in range(1, (_N - 1) // 3 + 1):
        a = 1
        while a + 3 * d <= _N:
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            a += 1
    return aps

_APS = _build_aps()
_POS_AP = [[] for _ in range(_N + 1)]
for _i, _ap in enumerate(_APS):
    for _p in _ap:
        _POS_AP[_p].append(_i)

def _is_valid_2coloring(c):
    for a, b, cc, d in _APS:
        if c[a] == c[b] == c[cc] == c[d]:
            return False
    return True

def _count_mono(c):
    m = 0
    for a, b, cc, d in _APS:
        if c[a] == c[b] == c[cc] == c[d]:
            m += 1
    return m

def _local_search(start, deadline):
    c = list(start)
    mono = _count_mono(c)
    while mono > 0 and time.time() < deadline:
        mono_aps = [(a, b, cc, d) for a, b, cc, d in _APS
                    if c[a] == c[b] == c[cc] == c[d]]
        if not mono_aps:
            break
        a, b, cc, d = random.choice(mono_aps)
        p = random.choice((a, b, cc, d))
        c[p] = 1 - c[p]
        mono = _count_mono(c)
    return c, mono

def _backtrack(deadline):
    c = [None] * (_N + 1)
    order = sorted(range(1, _N + 1), key=lambda p: -len(_POS_AP[p]))

    def unit_forced():
        for a, b, cc, d in _APS:
            vals = [c[a], c[b], c[cc], c[d]]
            un = [v for v in vals if v is None]
            if len(un) == 1:
                asg = [v for v in vals if v is not None]
                if asg[0] == asg[1] == asg[2]:
                    pos = (a, b, cc, d)[vals.index(None)]
                    return (pos, 1 - asg[0])
            elif len(un) == 0:
                if vals[0] == vals[1] == vals[2] == vals[3]:
                    return "CONTRADICTION"
        return None

    def search(k):
        if time.time() > deadline:
            return False
        while True:
            uf = unit_forced()
            if uf == "CONTRADICTION":
                return False
            if uf is None:
                break
            pos, fv = uf
            if c[pos] is not None:
                if c[pos] != fv:
                    return False
            else:
                c[pos] = fv
        pos = None
        for p in order:
            if c[p] is None:
                pos = p
                break
        if pos is None:
            return _is_valid_2coloring(c)
        for val in (0, 1):
            c[pos] = val
            if search(k + 1):
                return True
            c[pos] = None
        return False

    if search(0):
        return c
    return None

def _seed_fallback():
    colors = []
    for x in range(1, _N + 1):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                    for d in range(1, (x - 1) // 3 + 1))
                    and all(any(colors[x - 1 - j * d] != color for j in (1, 2))
                            for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Primary: find a 2-coloring (optimal, 2 distinct labels).
    for attempt in range(3000):
        if time.time() > deadline - 15:
            break
        start = [None] + [random.randint(0, 1) for _ in range(_N)]
        c, mono = _local_search(start, deadline - 15)
        if mono == 0 and _is_valid_2coloring(c):
            return [c[i] for i in range(1, _N + 1)]

    # Secondary: backtracking.
    if time.time() < deadline - 5:
        bc = _backtrack(deadline - 5)
        if bc is not None and _is_valid_2coloring(bc):
            return [bc[i] for i in range(1, _N + 1)]

    # Fallback: baseline seed construction.
    return _seed_fallback()
