import time
import math
import numpy as np
from scipy.optimize import minimize

DEADLINE = 150.0 - 5.0
START = time.time()
SQRT3 = math.sqrt(3.0)


def _time_left():
    return DEADLINE - (time.time() - START)


def _all_vertices(positions, ang_deg=0.0):
    a = math.radians(ang_deg)
    verts = []
    for (x, y) in positions:
        for k in range(6):
            t = a + k * math.pi / 3.0
            verts.append((x + math.cos(t), y + math.sin(t)))
    return np.array(verts)


def _min_side(verts, c, theta):
    ks = np.arange(6)
    nangs = theta + ks * (math.pi / 3.0)
    N = np.column_stack([np.cos(nangs), np.sin(nangs)])
    d = verts - np.array(c)
    proj = d @ N.T
    return (2.0 / SQRT3) * proj.max()


def _opt_outer(verts):
    best = (np.inf, None, None)
    for th0 in np.linspace(0.0, math.pi / 3.0, 21):
        def f(c):
            return _min_side(verts, c, th0)
        c0 = verts.mean(axis=0)
        res = minimize(f, c0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
        if res.fun < best[0]:
            best = (res.fun, res.x.copy(), th0)
    return best


def _no_overlap(pos, tol=1e-7):
    n = len(pos)
    lim = (SQRT3 - tol) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            if dx * dx + dy * dy < lim:
                return False
    return True


def _make_cluster(n=11):
    def ring(radius):
        if radius == 0:
            return [(0, 0)]
        res = []
        dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        q, r = radius, 0
        for di in range(6):
            for _ in range(radius):
                res.append((q, r))
                q += dirs[di][0]
                r += dirs[di][1]
        return res
    coords = [(0, 0)]
    rad = 1
    while len(coords) < n:
        coords.extend(ring(rad))
        rad += 1
    coords = coords[:n]
    pts = []
    for (q, r) in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return pts


def construct():
    pts = _make_cluster(11)
    cur = np.array(pts, dtype=float)
    verts = _all_vertices(cur)
    best_side, best_c, best_th = _opt_outer(verts)
    best_inner = [[float(x), float(y), 0.0] for (x, y) in cur]

    rng = np.random.default_rng(0)
    step = 0.4
    while _time_left() > 5.0:
        cand = cur.copy()
        idx = rng.integers(0, len(cand), size=max(1, len(cand) // 3))
        cand[idx] += rng.normal(0.0, step, size=(len(idx), 2))
        if _no_overlap(cand):
            s, c, th = _opt_outer(_all_vertices(cand))
            if s < best_side - 1e-9:
                best_side, best_c, best_th = s, c, th
                best_inner = [[float(x), float(y), 0.0] for (x, y) in cand]
                cur = cand
                step = min(step, 0.85)
            else:
                step *= 0.95
        else:
            step *= 0.9
        if step < 1e-4:
            step = 0.3
            cand = cur.copy()
            cand += rng.normal(0.0, 0.4, size=cand.shape)
            if _no_overlap(cand):
                s, c, th = _opt_outer(_all_vertices(cand))
                if s < best_side - 1e-9:
                    best_side, best_c, best_th = s, c, th
                    best_inner = [[float(x), float(y), 0.0] for (x, y) in cand]
                    cur = cand

    return best_inner, (float(best_c[0]), float(best_c[1])), float(best_side), float(math.degrees(best_th))
