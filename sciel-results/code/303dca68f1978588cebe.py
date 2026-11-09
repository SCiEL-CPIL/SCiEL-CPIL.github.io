"""Pack 11 unit regular hexagons into a minimum-side regular hexagon."""
import time
import math
import random

N = 11

_T0 = None

def _now():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def _unit_vec(a):
    c = math.cos(a)
    s = math.sin(a)
    return (c, s)

def _enclosing_side(pts, ang):
    # pts: list of (x,y) centers. Returns minimal outer side for orientation ang.
    d1 = _unit_vec(ang)
    d2 = _unit_vec(ang + math.pi / 3.0)
    d3 = _unit_vec(ang + 2.0 * math.pi / 3.0)
    best = 0.0
    for (x, y) in pts:
        for (dx, dy) in (d1, d2, d3):
            v = abs(x * dx + y * dy) + 2.0 / math.sqrt(3.0)
            if v > best:
                best = v
    return best

def _cost(pts, ang):
    s = _enclosing_side(pts, ang)
    return s * s

def _flower():
    # 1 center + 10 in a 3-2-3-3 ring (flat-top unit hexagons).
    r = math.sqrt(3.0)
    pts = [(0.0, 0.0)]
    ring = [
        (0.0, r), (r, r / 2.0), (-r, r / 2.0),
        (r, -r / 2.0), (-r, -r / 2.0),
        (2.0 * r, 0.0), (0.0, 0.0), (-2.0 * r, 0.0),
        (r, -r / 2.0) * 1.0, (-r, -r / 2.0) * 1.0,
    ]
    # Build a clean 10-ring: two offset rows of 3 + two of 2
    ring = []
    for k in range(-1, 2):
        ring.append((k * 2.0, r))
    for k in range(-1, 1):
        ring.append((k * 2.0 + 1.0, r / 2.0))
    for k in range(-1, 2):
        ring.append((k * 2.0, -r / 2.0))
    for k in range(-1, 1):
        ring.append((k * 2.0 + 1.0, -r))
    pts += ring
    return pts

def construct():
    random.seed(20240607)
    deadline = 120.0
    guard = deadline - 25.0   # safety margin

    pts = _flower()
    ang = 0.0
    best_pts = [list(p) for p in pts]
    best_ang = ang
    best_val = _cost(best_pts, best_ang)

    # Coarse rotation scan to get a good starting orientation
    for i in range(24):
        a = math.pi * i / 24.0
        v = _cost(pts, a)
        if v < best_val:
            best_val = v
            best_ang = a
            best_pts = [list(p) for p in pts]
    ang = best_ang

    # Local search / simulated annealing
    T = 0.05
    step = 0.5
    it = 0
    while _now() < guard:
        it += 1
        i = random.randrange(N)
        da = random.uniform(-0.12, 0.12)
        dx = random.gauss(0.0, step)
        dy = random.gauss(0.0, step)
        npts = [list(p) for p in pts]
        npts[i] = [npts[i][0] + dx, npts[i][1] + dy]
        nang = ang + da
        nc = _cost(npts, nang)
        if nc < best_val:
            best_val = nc
            best_pts = [list(p) for p in npts]
            best_ang = nang
        if nc < _cost(pts, ang) or random.random() < math.exp(-(nc - _cost(pts, ang)) / max(T, 1e-9)):
            pts = npts
            ang = nang
        T *= 0.999
        if T < 1e-5:
            T = 1e-5
        if it % 200 == 0 and step > 0.05:
            step *= 0.97

    # Final rotation polish
    for _ in range(3):
        lo, hi = best_ang - 0.1, best_ang + 0.1
        for _ in range(60):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if _cost(best_pts, m1) < _cost(best_pts, m2):
                hi = m2
            else:
                lo = m1
        best_ang = (lo + hi) / 2.0

    # Center the packing at origin
    cx = sum(p[0] for p in best_pts) / N
    cy = sum(p[1] for p in best_pts) / N
    inner = [[p[0] - cx, p[1] - cy, best_ang * 180.0 / math.pi] for p in best_pts]
    side = _enclosing_side([ (p[0]-cx, p[1]-cy) for p in best_pts ], best_ang)
    return inner, (0.0, 0.0), side, best_ang * 180.0 / math.pi
