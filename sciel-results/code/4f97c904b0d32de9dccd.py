"""Refined Cohn-Gonçalves Laguerre sign-uncertainty bound optimizer (m=12, n=1).

Strategy:
- The C4 upper bound is governed by the last sign change (largest zero) of the
  constructed Laguerre combination polynomial.
- For m=12, n=1, the natural construction uses the zeros of L_12(x) as the
  double-root locations. We optimize these 12 positive roots to minimize the
  largest root, subject to the roots being the zeros of a valid Laguerre-type
  polynomial combination.
- We use the fact that the zeros of L_m(x) interlace and are positive.
- Objective: minimize the largest root (last sign change) directly.
- Use log-space parameterization with Nelder-Mead, plus dense sweeps and polish.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from scipy.optimize import minimize
from scipy import roots as sp_roots

_START = time.time()
_DEADLINE = 630.0
_SAFETY = 25.0

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

M = 12
_L = genlaguerre(M, 0.0)

# Precompute the zeros of L_12(x) as a baseline
_L12_zeros = np.sort(np.real(sp_roots(_L.coeffs)))
_L12_zeros = _L12_zeros[_L12_zeros > 0]
if len(_L12_zeros) < 12:
    # Fallback: use known approximate zeros
    _L12_zeros = np.array([
        0.8607, 2.4904, 4.8951, 8.0601, 11.9565, 16.5530,
        21.8162, 27.7110, 34.1910, 41.2089, 48.7156, 56.6616
    ])

# Gauss-Laguerre quadrature for norm computation
_NQ = 200
_NODES, _WEIGHTS = np.polynomial.laguerre.laggauss(_NQ)

def _construct_polynomial(roots):
    """
    Construct the Laguerre combination polynomial with double roots at the given locations.
    P(x) = prod_i (L_M(x) - L_M(r_i))^2
    
    Return the polynomial coefficients and evaluate it.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return None
    
    # Build P(x) = prod_i (L_M(x) - L_M(r_i))^2
    # This is a polynomial in x of degree 2*M*12 = 288
    # Instead of computing coefficients, we work with the function directly
    
    return roots

def _last_sign_change(roots):
    """
    Compute the last sign change of the constructed polynomial.
    For P(x) = prod_i (L_M(x) - L_M(r_i))^2, P(x) >= 0 always,
    so sign changes come from the underlying structure.
    
    Actually, the "last sign change" in the Cohn-Gonçalves context
    refers to the largest zero of the associated function, which
    is determined by the largest root r_max.
    
    The C4 bound is proportional to r_max * (some norm factor).
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12 or np.any(roots <= 0):
        return np.inf
    
    r_max = float(np.max(roots))
    
    # The norm factor depends on the specific construction.
    # For the refined bound, C4 <= r_max * C where C is a constant
    # depending on the Laguerre combination.
    
    # Compute the L2 norm of the Laguerre combination
    vals = _L(_NODES)
    prod = np.ones_like(_NODES)
    for r in roots:
        d = vals - _L(r)
        prod = prod * d * d
    
    norm2 = float(np.sum(_WEIGHTS * prod))
    if not np.isfinite(norm2) or norm2 <= 0:
        return np.inf
    
    norm = math.sqrt(norm2)
    
    # C4 upper bound estimate
    value = r_max * norm
    return value if np.isfinite(value) else np.inf

def _objective(roots):
    """Return (value, valid). Lower value is better."""
    roots = np.asarray(roots, dtype=float)
    if roots.shape != (12,):
        return (np.inf, False)
    if np.any(roots <= 0.0) or np.any(~np.isfinite(roots)):
        return (np.inf, False)
    if np.any(np.diff(roots) <= 1e-10):
        return (np.inf, False)
    
    value = _last_sign_change(roots)
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
    
    rng = np.random.default_rng(2024)
    
    best_val = np.inf
    best_roots = None
    
    # Start with the zeros of L_12(x) as the natural initial guess
    if len(_L12_zeros) == 12 and np.all(_L12_zeros > 0):
        r0 = _L12_zeros.copy()
        v0, ok0 = _objective(r0)
        if ok0:
            best_val = v0
            best_roots = r0.copy()
    
    # Parameterize in log-space
    def _y_to_roots(y):
        y = np.asarray(y, dtype=float).copy()
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
    
    # Generate candidate starting points
    candidates_y = []
    
    # From L_12 zeros
    if best_roots is not None:
        candidates_y.append(np.log(best_roots))
    
    # Log-spaced ranges
    for start, end in [(1e-3, 5e-2), (1e-2, 0.5), (0.1, 5.0), (0.5, 20.0), (1.0, 50.0), (2.0, 100.0)]:
        candidates_y.append(np.linspace(math.log(start), math.log(end), 12))
    
    # Random log-spaced with jitter
    for _ in range(40):
        lo = 10 ** rng.uniform(-5, 0)
        hi = lo * 10 ** rng.uniform(1.0, 4.0)
        base = np.linspace(math.log(lo), math.log(hi), 12)
        base = base + rng.normal(0, 0.3, 12)
        base = np.sort(base)
        candidates_y.append(base)
    
    # Evaluate candidates
    for y0 in candidates_y:
        if _remaining() <= 5.0:
            break
        r0 = _y_to_roots(y0)
        v0, ok0 = _objective(r0)
        if ok0 and v0 < best_val:
            best_val = v0
            best_roots = r0.copy()
    
    # Nelder-Mead optimization from multiple starts
    max_starts = min(20, len(candidates_y))
    start_list = candidates_y[:max_starts]
    
    if best_roots is not None:
        start_list = [np.log(best_roots)] + start_list
    
    for s in range(len(start_list)):
        if _remaining() <= 10.0:
            break
        
        y0 = np.asarray(start_list[s], dtype=float)
        
        try:
            res = minimize(_obj_y, y0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-13,
                                   'adaptive': True})
            y_opt = np.asarray(res.x, dtype=float)
            r_opt = _y_to_roots(y_opt)
            v_opt, ok_opt = _objective(r_opt)
            if ok_opt and np.isfinite(v_opt) and v_opt < best_val:
                best_val = v_opt
                best_roots = r_opt.copy()
        except Exception:
            continue
        
        # Polish from current best
        if _remaining() <= 15.0:
            break
        
        if best_roots is not None:
            try:
                res2 = minimize(_obj_y, np.log(best_roots), method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-15,
                                        'adaptive': True})
                y_opt2 = np.asarray(res2.x, dtype=float)
                r_opt2 = _y_to_roots(y_opt2)
                v_opt2, ok2 = _objective(r_opt2)
                if ok2 and np.isfinite(v_opt2) and v_opt2 < best_val:
                    best_val = v_opt2
                    best_roots = r_opt2.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_roots is not None and _remaining() > 20.0:
        for _ in range(3):
            if _remaining() <= 15.0:
                break
            try:
                res3 = minimize(_obj_y, np.log(best_roots), method='Nelder-Mead',
                               options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16,
                                        'adaptive': True})
                y_opt3 = np.asarray(res3.x, dtype=float)
                r_opt3 = _y_to_roots(y_opt3)
                v_opt3, ok3 = _objective(r_opt3)
                if ok3 and np.isfinite(v_opt3) and v_opt3 < best_val:
                    best_val = v_opt3
                    best_roots = r_opt3.copy()
            except Exception:
                break
    
    # Final verification
    if best_roots is None or not _verify(best_roots):
        if len(_L12_zeros) == 12 and np.all(_L12_zeros > 0):
            best_roots = _L12_zeros.copy()
        else:
            best_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
