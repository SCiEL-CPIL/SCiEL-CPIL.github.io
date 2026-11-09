import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 110.0


def _rot(vx, vy, a):
    c, s = math.cos(a), math.sin(a)
    return (vx * c - vy * s, vx * s + vy * c)


_HEXV = [(math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))) for k in range(6)]


def _dist_seg_pt(p, a, b):
    px, py = p
    ax, ay = a
    abx, aby = b[0] - ax, b[1] - ay
    l2 = abx * abx + aby * aby
    t = 0.0
    if l2 > 1e-12:
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / l2))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def _hex_pts(c, ang):
    return [(_rot(vx, vy, ang)[0] + c[0], _rot(vx, vy, ang)[1] + c[1]) for (vx, vy) in _HEXV]


def _hex_dist(pi, pj):
    A = _hex_pts((pi[0], pi[1]), pi[2])
    B = _hex_pts((pj[0], pj[1]), pj[2])
    dmin = 1e9
    for i in range(6):
        dmin = min(dmin, _dist_seg_pt(A[i], B[i], B[(i + 1) % 6]))
        dmin = min(dmin, _dist_seg_pt(B[i], A[i], A[(i + 1) % 6]))
    return dmin


def _build(x):
    cx, cy, oang = x[0], x[1], x[2]
    angles = [(x[3 + 3 * i], x[4 + 3 * i], x[5 + 3 * i]) for i in range(N)]
    return cx, cy, oang, angles


def _outer_side(cx, cy, angles, oang):
    maxv = 0.0
    for (px, py, _) in angles:
        for f in range(6):
            th = math.radians(60 * f + 30 + oang)
            nx, ny = math.cos(th), math.sin(th)
            d = (px - cx) * nx + (py - cy) * ny
            if d < 0.0:
                d = 0.0
            if d + 1.0 > maxv:
                maxv = d + 1.0
    return maxv


def objective(x):
    cx, cy, oang, angles = _build(x)
    S = _outer_side(cx, cy, angles, oang)
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _hex_dist(angles[i], angles[j])
            if d < 0.0:
                pen += d * d
    return S + 1000.0 * pen


def _sym_config(pattern, scale, rot0):
    pts = []
    for (r, td) in pattern:
        th = math.radians(td) + rot0
        pts.append((r * math.cos(th), r * math.sin(th)))
    while len(pts) < N:
        th = math.radians(60 * len(pts)) + rot0
        pts.append((2.4 * math.cos(th), 2.4 * math.sin(th)))
    pts = pts[:N]
    x = [0.0, 0.0, 0.0]
    for (px, py) in pts:
        x.extend([px * scale, py * scale, 0.0])
    return x


def _starts():
    starts = []
    starts.append(_sym_config([(0, 0)] + [(2.0, 60 * k) for k in range(6)] + [(3.6, 30 + 60 * k) for k in range(5)], 1.0, 0.0))
    starts.append(_sym_config([(1.2, 60 * k) for k in range(6)] + [(-1.2, 60 * k) for k in range(6)], 1.0, 0.0))
    starts.append(_sym_config([(1.8, 0)] + [(3.06, 60 * k + 30) for k in range(6)] + [(3.8, 60 * k) for k in range(5)], 0.95, 0.0))
    rng = np.random.default_rng(7)
    for t in range(5):
        x = [0.0, 0.0, 0.0]
        for i in range(N):
            r = rng.uniform(0.5, 3.4)
            a = rng.uniform(0, 2 * math.pi)
            ang = float(rng.choice([0.0, math.pi / 6, math.pi / 3, math.pi / 2, 0.0]))
            x.extend([r * math.cos(a), r * math.sin(a), ang])
        starts.append(x)
    return starts


def construct():
    best_x = None
    best_val = float('inf')
    for x0 in _starts():
        if time.time() > DEADLINE - 25:
            break
        try:
            r = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 300, 'maxfun': 3000, 'gtol': 1e-7})
            if r.fun < best_val:
                best_val = r.fun
                best_x = r.x
        except Exception:
            pass
    if best_x is None:
        best_x = _starts()[0]

    if time.time() < DEADLINE - 15:
        try:
            r2 = minimize(objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
            if r2.fun < best_val:
                best_val = r2.fun
                best_x = r2.x
        except Exception:
            pass

    cx, cy, oang, angles = _build(best_x)
    S = _outer_side(cx, cy, angles, oang)
    inner = [[angles[i][0], angles[i][1], math.degrees(angles[i][2])] for i in range(N)]
    return inner, (cx, cy), S, math.degrees(oang)
