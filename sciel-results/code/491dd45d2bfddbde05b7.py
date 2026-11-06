"""Optimize 8 planar points to maximize product of pairwise distances at fixed diameter."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _log_prod_and_diam(pts):
    """Return (log product of 28 pairwise distances, max pairwise distance)."""
    d = pts[:, 0][:, None] - pts[:, 0][None, :]
    e = pts[:, 1][:, None] - pts[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    iu = np.triu_indices(8, 1)
    log_prod = np.sum(np.log(dist[iu]))
    diam = np.max(dist)
    return log_prod, diam

def _objective(x):
    """Negative log-product with a penalty for diameter > 2."""
    pts = x.reshape(8, 2)
    log_prod, diam = _log_prod_and_diam(pts)
    if diam > 2.0:
        # smooth-ish penalty: quadratic in excess
        excess = diam - 2.0
        return -(log_prod) + 1e4 * excess * excess
    return -log_prod

def _regular_octagon():
    from math import cos, sin, pi
    return np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)], dtype=float)

def _score_of(pts):
    """Compute the task score: Delta / n^n after scaling to diameter 2."""
    pts = np.asarray(pts, dtype=float)
    d = pts[:, 0][:, None] - pts[:, 0][None, :]
    e = pts[:, 1][:, None] - pts[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    iu = np.triu_indices(8, 1)
    diam = np.max(dist)
    if diam <= 1e-12:
        return 0.0
    # Scale so diameter == 2
    scale = 2.0 / diam
    prod_unscaled = np.prod(dist[iu])
    # product after scaling: each of 28 distances multiplied by scale
    prod_scaled = prod_unscaled * (scale ** 28)
    n = 8
    return prod_scaled / (n ** n)

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 120.0

    from scipy.optimize import minimize

    best_pts = _regular_octagon()
    best_score = _score_of(best_pts)
    best_x = best_pts.reshape(-1).copy()

    rng = random.Random(12345)
    np_rng = np.random.default_rng(7)

    def make_initial(scale, jitter):
        base = _regular_octagon()
        pts = base * scale
        if jitter > 0:
            pts = pts + np_rng.normal(0.0, jitter, size=(8, 2))
        return pts

    # Restart configurations: vary scale around 1 and add jitter
    attempts = [
        (1.0, 0.0),
        (1.0, 0.05),
        (1.0, 0.1),
        (0.8, 0.05),
        (1.2, 0.05),
        (0.6, 0.1),
        (1.4, 0.1),
        (1.0, 0.2),
        (0.5, 0.2),
        (1.5, 0.2),
    ]
    # Also add a few fully random points within a disk
    for _ in range(4):
        ang = np_rng.uniform(0, 2 * np.pi, 8)
        rad = np_rng.uniform(0.3, 1.2, 8)
        attempts.append(('random', rad, ang))

    for idx, att in enumerate(attempts):
        if _remaining() < 5.0:
            break
        if isinstance(att, tuple) and att[0] == 'random':
            _, rad, ang = att
            pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        else:
            scale, jitter = att
            pts = make_initial(scale, jitter)

        x0 = pts.reshape(-1)
        try:
            res = minimize(_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.x is not None:
                cand_pts = res.x.reshape(8, 2)
                # Verify distinctness and finite
                if np.all(np.isfinite(cand_pts)):
                    s = _score_of(cand_pts)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_pts = cand_pts
                        best_x = res.x.copy()
        except Exception:
            pass

    # Final polish on the best solution with tighter tolerances
    if _remaining() > 10.0:
        try:
            res = minimize(_objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-10})
            if res.x is not None:
                cand_pts = res.x.reshape(8, 2)
                if np.all(np.isfinite(cand_pts)):
                    s = _score_of(cand_pts)
                    if s > best_score:
                        best_pts = cand_pts
                        best_score = s
        except Exception:
            pass

    # Ensure distinct points
    out = best_pts.tolist()
    # Safety: if any two points are (nearly) identical, nudge
    for i in range(8):
        for j in range(i + 1, 8):
            dx = out[i][0] - out[j][0]
            dy = out[i][1] - out[j][1]
            if dx * dx + dy * dy < 1e-12:
                out[j][1] += 1e-6
    return out
# EVOLVE-BLOCK-END
