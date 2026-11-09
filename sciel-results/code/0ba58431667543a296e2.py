"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

construct() -> list of 12 positive double-root locations (m=12, n=1).
The score is the upper bound on C4 (lower is better).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_laguerre, laguerre
from scipy.optimize import minimize, differential_evolution

_M = 12
_N = 1
_DEADLINE = 630.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _C4(r):
    """Upper bound on C4 = product over double roots of (r_j + 1/r_j)."""
    r = np.asarray(r, dtype=float)
    if np.min(r) <= 0 or np.any(~np.isfinite(r)):
        return float("inf")
    return float(np.prod(r + 1.0 / r))

def _seed_roots():
    roots = roots_laguerre(_M)
    return sorted(roots.real.tolist())

def _valid(r):
    r = np.asarray(r, dtype=float)
    if len(r) != _M or np.min(r) <= 0 or np.any(~np.isfinite(r)):
        return False
    if np.any(np.diff(r) <= 1e-9):
        return False
    return True

def construct():
    # Deadline guard
    if _time_left() < 10:
        return _seed_roots()

    best_r = _seed_roots()
    best_score = _C4(best_r)

    def _obj(x):
        r = np.sort(np.exp(x))
        if not _valid(r):
            return 1e12
        return _C4(r)

    # Local refinement from seed (log-space for positivity)
    try:
        x0 = np.log(np.array(best_r, dtype=float))
        res = minimize(_obj, x0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'maxfun': 30000})
        r_cand = np.sort(np.exp(res.x))
        if _valid(r_cand):
            s = _C4(r_cand)
            if np.isfinite(s) and s < best_score:
                best_score, best_r = s, r_cand.tolist()
    except Exception:
        pass

    # Deadline guard
    if _time_left() < 40:
        return best_r

    # Global search with differential evolution (log-space)
    try:
        def _de_obj(y):
            if _time_left() < 5:
                return 1e12
            r = np.sort(np.exp(y))
            if not _valid(r):
                return 1e12
            return _C4(r)

        bounds = [(-3.0, 5.3)] * _M
        de_res = differential_evolution(
            _de_obj, bounds, seed=1, maxiter=80, popsize=10,
            tol=1e-8, polish=True, init='sobol', updating='deferred')
        r_cand = np.sort(np.exp(de_res.x))
        if _valid(r_cand):
            s = _C4(r_cand)
            if np.isfinite(s) and s < best_score:
                best_score, best_r = s, r_cand.tolist()
    except Exception:
        pass

    # Final local polish from best
    try:
        if _time_left() > 15:
            x0 = np.log(np.array(best_r, dtype=float))
            res = minimize(_obj, x0, method='L-BFGS-B',
                           options={'maxiter': 5000, 'maxfun': 50000})
            r_cand = np.sort(np.exp(res.x))
            if _valid(r_cand):
                s = _C4(r_cand)
                if np.isfinite(s) and s < best_score:
                    best_score, best_r = s, r_cand.tolist()
    except Exception:
        pass

    return best_r
# EVOLVE-BLOCK-END
