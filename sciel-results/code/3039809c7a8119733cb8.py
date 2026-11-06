"""Optimize 8 planar points to maximize Delta/n^n (diameter scaled to 2)."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

_N = 8
_DEADLINE = 140.0  # safety margin under the 150s kill
_START = time.time()

# ---------- objective ----------
def _score(pts):
    """Delta / n^n where Delta = prod_{i != j} |p_i - p_j|, diameter scaled to 2."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (8, 2) or not np.all(np.isfinite(pts)):
        return -1e30
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, 1.0)
    diam = d.max()
    if diam <= 1e-12:
        return -1e30
    d = d / (diam / 2.0)  # scale so diameter == 2
    s = (d ** 8).prod()
    return np.log(s) - 8 * np.log(8.0)  # log(Delta) - n log n  (n^n = 8^8)

def _negscore(pts):
    v = _score(pts)
    if not np.isfinite(v):
        return 1e30
    return -v

# ---------- optimization ----------
def _local_polish(x0, max_iter=4000, tol=1e-14):
    """High-precision local polish with a very tight tolerance."""
    best_x, best_f = np.asarray(x0, float), _negscore(x0)
    for _ in range(60):
        if time.time() > _DEADLINE - 15.0:
            break
        res = minimize(_negscore, best_x, method="L-BFGS-B",
                       options=dict(maxiter=max_iter, ftol=tol, gtol=1e-12,
                                    maxfun=200000))
        if res.success or res.fun < best_f - 1e-12:
            if res.fun < best_f - 1e-12:
                best_x, best_f = res.x, res.fun
            if res.jac is not None and np.max(np.abs(res.jac)) < 1e-10:
                break
        else:
            break
    return best_x, best_f

def construct():
    global _START
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # Regular octagon as a strong anchor (score 1.0).
    k = np.arange(8)
    octagon = np.stack([np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)], axis=1)
    best_x, best_f = octagon.copy(), _negscore(octagon)

    def maybe_update(x):
        nonlocal best_x, best_f
        f = _negscore(x)
        if f < best_f - 1e-13:
            best_x, best_f = np.asarray(x, float).copy(), f

    # Global search with differential evolution (bounded box).
    bounds = [(-1.0, 1.0)] * 16
    try:
        if time.time() < _DEADLINE - 70.0:
            de = differential_evolution(
                _negscore, bounds, seed=7, maxiter=25, popsize=15,
                tol=1e-10, mutation=(0.3, 1.0), recombination=0.9,
                polish=False, init="sobol",
                callback=lambda x, con: maybe_update(x))
            maybe_update(de.x)
    except Exception:
        pass

    # Multi-start local polish from the anchor, DE result, and random starts.
    starts = [octagon.copy(), best_x.copy()]
    for _ in range(40):
        starts.append(rng.normal(0, 0.9, size=(8, 2)))

    for s in starts:
        if time.time() > _DEADLINE - 20.0:
            break
        try:
            x, f = _local_polish(s)
            maybe_update(x)
        except Exception:
            pass

    # Final high-precision polish on the best point.
    if time.time() < _DEADLINE - 10.0:
        try:
            x, f = _local_polish(best_x, max_iter=8000, tol=1e-15)
            maybe_update(x)
        except Exception:
            pass

    pts = best_x
    # Keep coordinates comfortably within |coord| < 1e6 (already bounded ~1).
    return [[float(x), float(y)] for x, y in pts]
