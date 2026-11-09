"""Structural change: use roots of L_12^(0)(x) as baseline, then apply coordinate descent.

Strategy:
- Initialize with the 12 positive roots of the generalized Laguerre polynomial L_12^(0)(x).
- These roots are known to minimize the weighted L2 norm and provide a physically consistent baseline.
- Apply sequential coordinate descent in log-space to refine each root individually.
- Use a finer Gauss-Laguerre quadrature for better accuracy.
- Explicit deadline guard: return the best result found before the 630 s limit.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize_scalar

_START = time.time()
_DEADLINE = 630.0
_SAFETY = 25.0  # seconds to stop before the hard kill

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

M = 12
_L = genlaguerre(M, 0.0)

# Precompute Gauss-Laguerre quadrature (alpha=0) for the L2 norm on [0, inf)
_NQ = 500
_NODES, _WEIGHTS = np.polynomial.laguerre.laggauss(_NQ)

def _get_laguerre_roots():
    """Get the 12 positive roots of L_12^(0)(x)."""
    # Use numpy to find roots of the Laguerre polynomial
    coeffs = _L.coeffs
    roots = np.roots(coeffs)
    # Filter to positive real roots
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    positive_roots = real_roots[real_roots > 0]
    positive_roots = np.sort(positive_roots)
    return positive_roots

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

def _clamp_roots(roots, min_gap=1e-8):
    """Ensure roots are positive, ordered, and have minimum gap."""
    roots = np.asarray(roots, dtype=float).copy()
    roots = np.sort(roots)
    for i in range(1, 12):
        if roots[i] < roots[i-1] + min_gap:
            roots[i] = roots[i-1] + min_gap
    roots = np.clip(roots, 1e-9, 1e6)
    return roots

def construct():
    """Return list of 12 positive double-root locations (m=12, n=1).
    Score is the upper bound on C4 (lower is better)."""
    # Start with the roots of L_12^(0)(x)
    initial_roots = _get_laguerre_roots()
    if len(initial_roots) < 12:
        # Fallback if something goes wrong
        initial_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    best_roots = initial_roots.copy()
    best_val, _ = _objective(best_roots)
    
    # Coordinate descent in log-space
    # For each root, optimize it while keeping others fixed
    max_iterations = 50
    converged = False
    
    for iteration in range(max_iterations):
        if _remaining() <= 10.0:
            break
        
        improved = False
        for i in range(12):
            if _remaining() <= 5.0:
                break
            
            # Save current root
            orig_root = best_roots[i]
            
            # Define objective for single root optimization
            def _single_obj(log_r):
                r = math.exp(log_r)
                new_roots = best_roots.copy()
                new_roots[i] = r
                # Ensure ordering
                if i > 0 and r <= new_roots[i-1] + 1e-8:
                    return 1e12
                if i < 11 and r >= new_roots[i+1] - 1e-8:
                    return 1e12
                if r <= 0:
                    return 1e12
                
                v, ok = _objective(new_roots)
                if not ok:
                    return 1e12
                return v
            
            # Determine search bounds for log_r
            if i == 0:
                log_lo = math.log(max(1e-9, best_roots[0] * 0.1))
                log_hi = math.log(min(1e6, best_roots[1] * 10 if len(best_roots) > 1 else 1e6))
            elif i == 11:
                log_lo = math.log(max(1e-9, best_roots[10] * 0.1 if len(best_roots) > 10 else 1e-9))
                log_hi = math.log(1e6)
            else:
                log_lo = math.log(max(1e-9, best_roots[i-1] * 0.1))
                log_hi = math.log(min(1e6, best_roots[i+1] * 10))
            
            # Add margin
            log_lo -= 0.5
            log_hi += 0.5
            
            try:
                res = minimize_scalar(_single_obj, bounds=(log_lo, log_hi), method='bounded',
                                      options={'maxiter': 200, 'xatol': 1e-12})
                new_log_r = res.x
                new_r = math.exp(new_log_r)
                
                # Check if this improves
                new_roots = best_roots.copy()
                new_roots[i] = new_r
                new_roots = _clamp_roots(new_roots)
                
                v_new, ok_new = _objective(new_roots)
                if ok_new and v_new < best_val - 1e-12:
                    best_roots = new_roots
                    best_val = v_new
                    improved = True
            except Exception:
                continue
        
        if not improved:
            converged = True
            break
    
    # Final verification
    if not _verify(best_roots):
        best_roots = initial_roots.copy()
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
