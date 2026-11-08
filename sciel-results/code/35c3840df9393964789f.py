"""Heilbronn triangles: maximize the minimum triangle area for 11 points in a unit-area
equilateral triangle (0,0),(1,0),(1/2,sqrt(3)/2)."""
import numpy as np
import random
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
V_A = np.array([0.0, 0.0])
V_B = np.array([1.0, 0.0])
V_C = np.array([0.5, SQRT3 / 2.0])
VERTS = np.array([V_A, V_B, V_C])
EDGE_MIDS = np.array([
    (V_A + V_B) / 2.0,
    (V_B + V_C) / 2.0,
    (V_C + V_A) / 2.0,
])
CENTROID = (V_A + V_B + V_C) / 3.0
DEADLINE = time.time() + 110.0  # safety margin under 120 s / 150 s wall clock


def _barycentric_inside(p):
    """Return True if point p is on or inside the reference triangle."""
    v0 = V_B - V_A
    v1 = V_C - V_A
    v2 = p - V_A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return False
    v = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    return (v >= -1e-9) and (w >= -1e-9) and (v + w <= 1.0 + 1e-9)


def _clip_to_triangle(p):
    """Clamp a point to the triangle by barycentric projection."""
    v0 = V_B - V_A
    v1 = V_C - V_A
    v2 = p - V_A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return p.copy()
    v = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    if v < 0.0:
        v = 0.0
    if w < 0.0:
        w = 0.0
    if v + w > 1.0:
        s = v + w
        v /= s
        w /= s
    return V_A + v * v0 + w * v1


def _min_area(pts):
    """Vectorized minimum triangle area over all triples of points."""
    n = pts.shape[0]
    best = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            rest = pts[j + 1:]
            if rest.shape[0] == 0:
                continue
            # area = 0.5 * |cross(d, pts[k]-pts[i])|
            cross = d[0] * (rest[:, 1] - pts[i, 1]) - d[1] * (rest[:, 0] - pts[i, 0])
            areas = 0.5 * np.abs(cross)
            m = areas.min()
            if m < best:
                best = m
    return best


def _is_valid(pts):
    if pts.shape != (N, 2):
        return False
    for p in pts:
        if not _barycentric_inside(p):
            return False
    return True


def _structured_start():
    """A reasonable deterministic configuration: 3 vertices, 3 edge midpoints, 5 interior."""
    pts = []
    pts.append(V_A.copy())
    pts.append(V_B.copy())
    pts.append(V_C.copy())
    pts.append(EDGE_MIDS[0].copy())
    pts.append(EDGE_MIDS[1].copy())
    pts.append(EDGE_MIDS[2].copy())
    # 5 interior points: centroid + a small ring around it
    pts.append(CENTROID.copy())
    ring_r = 0.18
    for ang in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
        # offset in triangle coordinate space, then check inside
        cand = CENTROID + ring_r * np.array([np.cos(ang), np.sin(ang) * (SQRT3 / 2.0) * 1.0])
        # use a simpler interior placement: scaled toward centroid from vertices
        cand = CENTROID + 0.22 * np.array([np.cos(ang), np.sin(ang)])
        cand = _clip_to_triangle(cand)
        pts.append(cand)
    pts = np.array(pts)
    return pts


def _random_start(rng):
    pts = []
    # keep a few structural anchors to avoid terrible clustering
    pts.append(_clip_to_triangle(V_A + 0.02 * np.array([rng.random(), rng.random()])))
    pts.append(_clip_to_triangle(V_B + 0.02 * np.array([-rng.random(), rng.random()])))
    pts.append(_clip_to_triangle(V_C + 0.02 * np.array([rng.random() - 0.5, -rng.random()])))
    while len(pts) < N:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V_A + u * (V_B - V_A) + v * (V_C - V_A)
            pts.append(p)
    return np.array(pts)


def _coordinate_descent(pts, max_iter=400, step=0.15, rng=None):
    """Improve by greedily moving one point at a time to increase the min area."""
    cur = _min_area(pts)
    for it in range(max_iter):
        improved = False
        for i in range(N):
            orig = pts[i].copy()
            # try a set of candidate moves
            candidates = [orig + step * np.array([np.cos(a), np.sin(a)]) for a in
                          np.linspace(0, 2 * np.pi, 8, endpoint=False)]
            # also try smaller steps
            for c in candidates:
                c = _clip_to_triangle(c)
                pts[i] = c
                v = _min_area(pts)
                if v > cur + 1e-12:
                    cur = v
                    improved = True
                    break
            if not improved:
                pts[i] = orig
            else:
                improved = False
        if not improved:
            break
    return pts, cur


def _simulated_annealing(pts, t0=0.05, t1=0.001, iters=1500, rng=None):
    cur = _min_area(pts)
    best_pts = pts.copy()
    best_val = cur
    T = t0
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / max(1, iters - 1))
        i = rng.randrange(N)
        orig = pts[i].copy()
        ang = rng.random() * 2 * np.pi
        r = rng.random() * 0.12
        cand = orig + r * np.array([np.cos(ang), np.sin(ang)])
        cand = _clip_to_triangle(cand)
        pts[i] = cand
        v = _min_area(pts)
        if v > cur or rng.random() < np.exp((v - cur) / max(T, 1e-9)):
            cur = v
            if v > best_val:
                best_val = v
                best_pts = pts.copy()
        else:
            pts[i] = orig
    return best_pts, best_val


def construct():
    global DEADLINE
    start = time.time()
    rng = random.Random(12345)

    best_pts = None
    best_val = -1.0

    def consider(pts):
        nonlocal best_pts, best_val
        if not _is_valid(pts):
            return
        v = _min_area(pts)
        if v > best_val:
            best_val = v
            best_pts = pts.copy()

    # 1) structured start + coordinate descent + SA
    s0 = _structured_start()
    consider(s0)
    cd_pts, _ = _coordinate_descent(s0, max_iter=300, step=0.12, rng=rng)
    consider(cd_pts)
    sa_pts, _ = _simulated_annealing(cd_pts, t0=0.04, t1=0.0005, iters=1200, rng=rng)
    consider(sa_pts)

    # 2) random restarts
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > DEADLINE - 15.0:
            break
        rp = _random_start(rng)
        consider(rp)
        cd_pts, _ = _coordinate_descent(rp, max_iter=250, step=0.10, rng=rng)
        consider(cd_pts)
        if time.time() > DEADLINE - 15.0:
            break
        sa_pts, _ = _simulated_annealing(cd_pts, t0=0.03, t1=0.0004, iters=1000, rng=rng)
        consider(sa_pts)

    # 3) final polish on the best found
    if best_pts is not None and time.time() < DEADLINE - 10.0:
        polish, _ = _coordinate_descent(best_pts, max_iter=400, step=0.05, rng=rng)
        consider(polish)
        if time.time() < DEADLINE - 10.0:
            polish2, _ = _simulated_annealing(polish, t0=0.02, t1=0.0002, iters=800, rng=rng)
            consider(polish2)

    if best_pts is None:
        # fallback: structured start
        best_pts = _structured_start()

    # enforce strict validity
    if not _is_valid(best_pts):
        fixed = np.array([_clip_to_triangle(p) for p in best_pts])
        if _is_valid(fixed):
            best_pts = fixed
        else:
            best_pts = _structured_start()

    return best_pts
