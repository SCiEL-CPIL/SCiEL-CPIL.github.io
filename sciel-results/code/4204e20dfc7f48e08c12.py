"""Honeycomb-cluster design: fixed dense 12-hexagon packing, optimize orientation + position."""
import math
import time
import numpy as np

N = 12
DEADLINE = 140.0          # explicit wall-clock safety margin before the 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

# ---------- geometry ----------
# Flat-top unit hexagon vertices (side 1, centered at origin)
_HEX_V = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# Dense honeycomb cluster: center + ring of 6 + ring of 5 (skip one gap to keep 12)
_RING2 = [(1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)]   # 5 of the 6 second-ring cells
_CLUSTER = [(0, 0)] + [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)] + _RING2
assert len(_CLUSTER) == 12

def _cluster_verts():
    pts = []
    for cx, cy in _CLUSTER:
        ox = 1.5 * cx
        oy = math.sqrt(3.0) / 2.0 * (3.0 * cy + cx)
        pts.append(_HEX_V + np.array([ox, oy]))
    return np.array(pts)

_PTS = _cluster_verts()                      # all 72 vertices in cluster frame
_PTS_C = _PTS - _PTS.mean(axis=0)            # centered on centroid (translation-invariant)

def _axes(theta):
    """Three support axes of the enclosing hexagon, rotated by theta."""
    c, s = math.cos(theta), math.sin(theta)
    u = np.array([c, s])
    v = np.array([c * math.cos(math.radians(60)) - s * math.sin(math.radians(60)),
                  c * math.sin(math.radians(60)) + s * math.cos(math.radians(60))])
    w = np.array([c * math.cos(math.radians(120)) - s * math.sin(math.radians(120)),
                  c * math.sin(math.radians(120)) + s * math.cos(math.radians(120))])
    return np.stack([u, v, w])

def _L(theta, cxy):
    """Minimal enclosing-hexagon side length for cluster rotated by theta, centered at cxy."""
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    P = (_PTS_C @ rot.T) + cxy
    A = _axes(theta)
    d = P @ A.T
    return float(np.max(np.abs(d)))

def _best_center(theta):
    """Optimal translation for a given orientation: center = -centroid of the 6 extreme points."""
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    Q = _PTS_C @ rot.T
    A = _axes(theta)
    d = Q @ A.T
    idx = np.zeros((len(Q), 6), dtype=int)
    for k in range(6):
        if k < 3:
            idx[:, k] = np.argmin(d[:, k])
        else:
            idx[:, k] = np.argmax(d[:, k - 3])
    return -Q[idx].mean(axis=0)

def _result(theta, cxy, L):
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    inner = []
    for cx, cy in _CLUSTER:
        ox = 1.5 * cx
        oy = math.sqrt(3.0) / 2.0 * (3.0 * cy + cx)
        p = rot @ np.array([ox, oy]) + cxy
        inner.append([float(p[0]), float(p[1]), 0.0])
    return inner, (float(cxy[0]), float(cxy[1])), float(L), 0.0

def _refine(theta0, cxy0, L0):
    """Local Nelder-Mead refinement over (theta, cx, cy)."""
    from scipy.optimize import minimize
    best = [L0]
    def f(x):
        th = float(x[0])
        L = _L(th, np.array(x[1:3]))
        if L < best[0]:
            best[0] = L
        return L
    x0 = np.array([theta0, cxy0[0], cxy0[1]])
    try:
        res = minimize(f, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-13, "fatol": 1e-13,
                                "adaptive": True, "initial_simplex": [0.02, 0.02, 0.02]})
        if res.fun < best[0]:
            best[0] = float(res.fun)
    except Exception:
        pass
    return best[0]

def construct():
    # 1) coarse orientation grid, optimal center each time
    best = (None, None, 1e18)
    for i in range(0, 360, 2):
        if _elapsed() > DEADLINE:
            break
        th = math.radians(i)
        c = _best_center(th)
        L = _L(th, c)
        if L < best[2]:
            best = (th, c, L)
    if best[0] is None:
        th, c, L = 0.0, np.zeros(2), _L(0.0, np.zeros(2))
    else:
        th, c, L = best

    # 2) fine orientation grid around the best coarse angle
    for i in range(-15, 16, 1):
        if _elapsed() > DEADLINE:
            break
        th2 = th + math.radians(i)
        c2 = _best_center(th2)
        L2 = _L(th2, c2)
        if L2 < L:
            th, c, L = th2, c2, L2

    # 3) ultra-fine orientation sweep
    for i in range(-20, 21, 1):
        if _elapsed() > DEADLINE:
            break
        th2 = th + i * (2.0 / 180.0)
        c2 = _best_center(th2)
        L2 = _L(th2, c2)
        if L2 < L:
            th, c, L = th2, c2, L2

    # 4) local polish (orientation + translation)
    if _elapsed() < DEADLINE:
        L = _refine(th, c, L)
        c = _best_center(th)
        L = _L(th, c)

    return _result(th, c, L)
