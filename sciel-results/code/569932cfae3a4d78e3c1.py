"""Rigid honeycomb cluster: optimize global rotation + translation via grid search."""
import math
import time

import numpy as np

N = 12
DEADLINE = 150.0
_SAFE = 12.0
_T = 1.0 / math.sqrt(3.0)      # 1/sqrt(3)
_R = math.sqrt(3.0) / 2.0      # inradius = sqrt(3)/2
_SQ3 = math.sqrt(3.0)

# ---- honeycomb shapes: 12 hexagons as a rigid cluster (centers in unit spacing) ----
def _shape_honeycomb():
    pts = []
    for r in range(-1, 2):
        for c in range(-1, 2):
            pts.append((c + 0.5 * r, _SQ3 * r))
    return np.array(pts)

def _shape_crowded():
    pts = []
    for r in range(-1, 2):
        for c in range(-2, 2):
            pts.append((c + 0.5 * r, _SQ3 * r))
    return np.array(pts)

def _shape_line():
    pts = []
    for i in range(-3, 4):
        pts.append((2.0 * i, 0.0))
    return np.array(pts)

_SHAPES = [_shape_honeycomb(), _shape_crowded(), _shape_line()]

# ---- geometry ----
def _normals(theta):
    a = np.array([0.0, _SQ3, _SQ3, 0.0, -_SQ3, -_SQ3])
    b = np.array([-_SQ3, 0.0, _SQ3, _SQ3, 0.0, -_SQ3])
    c, s = math.cos(theta), math.sin(theta)
    M = np.array([[c, -s], [s, c]])
    return M @ np.stack([a, b], axis=1)   # (6,2)

def _side_from(theta, P, c):
    U = _normals(theta)
    d = (P - c) @ U.T                     # (n,6)
    return (np.abs(d).max() + 1.0) * _T

def _side_for_shape(theta, P, c):
    return _side_from(theta, P, c)

def _best_side_for_shape(theta, P, c):
    # minimize over translation c  ->  max over points of max over normals
    U = _normals(theta)
    d = P @ U.T                            # (n,6)
    m = np.max(np.abs(d), axis=0)          # (6,)
    return (m.max() + 1.0) * _T

def _optimal_translation(theta, P):
    # c that minimizes the bounding side (Chebyshev center in normal space)
    U = _normals(theta)
    d = P @ U.T
    lo = d.min(axis=0)
    hi = d.max(axis=0)
    return U @ (0.5 * (lo + hi))

# ---- grid search over rotation ----
def _search_shape(P, t_start, budget):
    # coarse
    best_s = float('inf'); best_theta = 0.0; best_c = np.zeros(2)
    step = 0.02
    for i in range(0, 180, int(step * 180.0) or 1):
        theta = i * math.pi / 180.0
        s = _best_side_for_shape(theta, P)
        if s < best_s:
            best_s = s; best_theta = theta
        if time.time() > t_start + budget:
            break
    # refine around best
    for span, st in [(0.10, 0.002), (0.02, 0.0004)]:
        lo, hi = best_theta - span, best_theta + span
        n = max(2, int((hi - lo) / st))
        for j in range(n + 1):
            theta = lo + (hi - lo) * j / n
            s = _best_side_for_shape(theta, P)
            if s < best_s:
                best_s = s; best_theta = theta
        if time.time() > t_start + budget:
            break
    best_c = _optimal_translation(best_theta, P)
    return best_s, best_theta, best_c

def _to_inner(P, theta, c):
    ct, st = math.cos(theta), math.sin(theta)
    M = np.array([[ct, -st], [st, ct]])
    Q = P @ M.T + c
    deg = math.degrees(theta)
    return [[float(Q[i, 0]), float(Q[i, 1]), float(deg)] for i in range(len(Q))]

def construct():
    t_start = time.time()
    budget = DEADLINE - _SAFE
    best = None
    for P in _SHAPES:
        if time.time() - t_start > budget:
            break
        s, theta, c = _search_shape(P, t_start, budget - (time.time() - t_start))
        if best is None or s < best[0]:
            best = (s, theta, c, P)
    if best is None:
        s, theta, c, P = _search_shape(_SHAPES[0], t_start, budget)
    else:
        s, theta, c, P = best
    inner = _to_inner(P, theta, c)
    return inner, (float(c[0]), float(c[1])), float(s), float(theta)
