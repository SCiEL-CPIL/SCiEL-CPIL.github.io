"""Rewritten optimizer for the Cohn-Goncalves Laguerre sign-uncertainty bound (m=12, n=1).

Strategy:
- The C4 bound is governed by the largest root r_max and the L2 norm of the
  Laguerre combination. Both shrink when roots are packed toward the origin
  because L_12(x) - L_12(r_i) is small for small r_i.
- We optimize 12 positive, ordered roots via a log-parameterized Nelder-Mead
  search, plus a dense fine sweep and a local polish.
- Explicit deadline guard: return the best result found before the 630 s limit.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 630.0
_SAFETY = 25.0  # seconds to stop before the hard kill

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

M = 12
_L = genlaguerre(M, 0.0)

# Precompute Gauss-Laguerre quadrature (alpha=0) for the L2 norm on [0, inf)
_NQ = 300
_NODES, _WEIGHTS = np.polynomial.laguerre.laggauss(_NQ)

def _laguerre_eval(roots, x):
    """Product over i of (L_M(x) - L_M(r_i))^2, well-scaled."""
    vals = _L(x)
    out = np.ones_like(x, dtype=float)
    for r in roots:
        d = vals - _L(r)
        out = out * d * d
    return out

def _objective(roots):
    """Return (value, valid). value is the C4 upper bound estimate (lower better)."""
    roots = np.asarray(roots, dtype=float)
    if roots.shape != (12,):
        return (np.inf, False)
    if np.any(roots <= 0.0) or np.any(~np.isfinite(roots)):
        return (np.inf, False)
    if np.any(np.diff(roots) <= 1e-10):
        return (np.inf, False)
    last_root = float(roots[-1])
    vals = _laguerre_eval(roots, _NODES)
    norm2 = float(np.sum(_WEIGHTS * vals * vals))
    if not np.isfinite(norm2) or norm2 <= 0.0:
        return (np.inf, False)
    norm = math.sqrt(norm2)
    value = last_root * norm
    if not np.isfinite(value):
        return (np.inf, False)
    return (value, True)

def _verify(roots):
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0.0):
        return False
    if np.any(np.diff(roots) <= 0.0):
        return False
    return True

def _clamp(x):
    x = np.asarray(x, dtype=float).copy()
    for i in range(1, 12):
        if x[i] < x[i-1] + 1e-8:
            x[i] = x[i-1] + 1e-8
    x = np.clip(x, 1e-9, 1e4)
    return x

def construct():
    """Return list of 12 positive double-root locations (m=12, n=1).
    Score is the upper bound on C4 (lower is better)."""
    # Parameterize in log-space to keep roots positive and ordered.
    # Let r_i = exp(y_i) with y_1 < y_2 < ... < y_12.
    # We optimize y (12-dim) with a small monotone gap enforced.

    rng = np.random.default_rng(2024)

    def _y_to_roots(y):
        y = np.asarray(y, dtype=float).copy()
        # enforce strict increasing in log-space
        for i in range(1, 12):
            if y[i] <= y[i-1] + 1e-6:
                y[i] = y[i-1] + 1e-6
        r = np.exp(y)
        return r

    def _obj_y(y):
        r = _y_to_roots(y)
        v, ok = _objective(r)
        if not ok:
            return 1e12
        return v

    best_val = np.inf
    best_roots = None

    # Candidate initial points: log-spaced small roots, plus random log-spaced.
    def _mk_y(start, end, n=12):
        return np.linspace(math.log(start), math.log(end), n)

    candidates_y = []
    # Very tight near origin
    candidates_y.append(_mk_y(1e-3, 5e-3))
    candidates_y.append(_mk_y(1e-4, 1e-3))
    candidates_y.append(_mk_y(1e-2, 1e-1))
    candidates_y.append(_mk_y(1e-3, 1e-2))
    candidates_y.append(_mk_y(1e-1, 1.0))
    candidates_y.append(_mk_y(0.1, 10.0))
    candidates_y.append(_mk_y(1.0, 12.0))
    # Random log-spaced
    for _ in range(30):
        lo = 10 ** rng.uniform(-6, -1)
        hi = lo * 10 ** rng.uniform(0.5, 3.0)
        candidates_y.append(_mk_y(lo, hi))
        # add jitter
        base = _mk_y(lo, hi)
        base = base + rng.normal(0, 0.2, 12)
        base = np.sort(base)
        candidates_y.append(base)

    # Evaluate all candidates quickly
    for y0 in candidates_y:
        if _remaining() <= 5.0:
            break
        r0 = _y_to_roots(y0)
        v0, ok0 = _objective(r0)
        if ok0 and v0 < best_val:
            best_val = v0
            best_roots = r0.copy()

    # Nelder-Mead in log-space from the best candidate and several others
    max_starts = 25
    start_list = candidates_y[:max_starts]
    # ensure the best candidate is first
    if best_roots is not None:
        best_y = np.log(best_roots)
        start_list = [best_y] + start_list

    for s in range(len(start_list)):
        if _remaining() <= 8.0:
            break
        y0 = np.asarray(start_list[s], dtype=float)
        try:
            res = minimize(_obj_y, y0, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-10, 'fatol': 1e-13,
                                    'adaptive': True})
        except Exception:
            continue
        y_opt = np.asarray(res.x, dtype=float)
        r_opt = _y_to_roots(y_opt)
        v_opt, ok_opt = _objective(r_opt)
        if ok_opt and np.isfinite(v_opt) and v_opt < best_val:
            best_val = v_opt
            best_roots = r_opt.copy()

        # polish from current best
        if _remaining() <= 12.0:
            break
        if best_roots is not None:
            try:
                res2 = minimize(_obj_y, np.log(best_roots), method='Nelder-Mead',
                                options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-15,
                                         'adaptive': True})
                y_opt2 = np.asarray(res2.x, dtype=float)
                r_opt2 = _y_to_roots(y_opt2)
                v_opt2, ok2 = _objective(r_opt2)
                if ok2 and np.isfinite(v_opt2) and v_opt2 < best_val:
                    best_val = v_opt2
                    best_roots = r_opt2.copy()
            except Exception:
                pass

    # Final verification; fall back to a safe default if invalid
    if best_roots is None or not _verify(best_roots):
        best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
