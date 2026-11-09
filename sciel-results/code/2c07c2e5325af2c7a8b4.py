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
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 630.0
_SAFETY = 25.0

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

M = 12
_L = genlaguerre(M, 0.0)

# Precompute Gauss-Laguerre quadrature nodes and weights
_NQ = 400
_NODES, _WEIGHTS = np.polynomial.laguerre.laggauss(_NQ)

def _get_laguerre_roots():
    """Get the 12 positive roots of L_12^(0)(x)."""
    coeffs = _L.coeffs
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    positive_roots = real_roots[real_roots > 0]
    positive_roots = np.sort(positive_roots)
    return positive_roots

def _laguerre_eval(roots, x):
    """Product over i of (L_M(x) - L_M(r_i))^2."""
    vals = _L(x)
    out = np.ones_like(x, dtype=float)
    for r in roots:
        d = vals - _L(r)
        out = out * d * d
    return out

def _objective(roots):
    """Return the C4 upper bound estimate (lower is better)."""
    roots = np.asarray(roots, dtype=float)
    if roots.shape != (12,):
        return np.inf
    if np.any(roots <= 0.0) or np.any(~np.isfinite(roots)):
        return np.inf
    if np.any(np.diff(roots) <= 1e-10):
        return np.inf
    last_root = float(roots[-1])
    vals = _laguerre_eval(roots, _NODES)
    norm2 = float(np.sum(_WEIGHTS * vals * vals))
    if not np.isfinite(norm2) or norm2 <= 0.0:
        return np.inf
    norm = math.sqrt(norm2)
    value = last_root * norm
    if not np.isfinite(value):
        return np.inf
    return value

def _verify(roots):
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0.0):
        return False
    if np.any(np.diff(roots) <= 0.0):
        return False
    return True

def _log_objective(log_roots):
    """Objective in log-space: exp(log_roots) are the actual roots."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0.0) or np.any(~np.isfinite(roots)):
        return 1e12
    if np.any(np.diff(roots) <= 1e-8):
        # Penalty for violating ordering/gap
        return 1e12 + 1e6 * np.sum(np.maximum(0, 1e-8 - np.diff(roots)))
    return _objective(roots)

def _log_objective_grad(log_roots):
    """Gradient of the log-space objective (numerical)."""
    eps = 1e-7
    grad = np.zeros(12)
    for i in range(12):
        lp = log_roots.copy()
        lm = log_roots.copy()
        lp[i] += eps
        lm[i] -= eps
        grad[i] = (_log_objective(lp) - _log_objective(lm)) / (2 * eps)
    return grad

def construct():
    """Return list of 12 positive double-root locations (m=12, n=1).
    Score is the upper bound on C4 (lower is better)."""
    # Start with the roots of L_12^(0)(x)
    initial_roots = _get_laguerre_roots()
    if len(initial_roots) < 12:
        initial_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=float)
    
    # Convert to log-space
    initial_log_roots = np.log(initial_roots)
    
    best_roots = initial_roots.copy()
    best_val = _objective(best_roots)
    
    # Multi-start with L-BFGS-B in log-space
    methods = ['L-BFGS-B', 'BFGS', 'Nelder-Mead']
    
    for method in methods:
        if _remaining() <= 30.0:
            break
        
        # Try multiple random perturbations as starting points
        n_starts = 5
        for start_idx in range(n_starts):
            if _remaining() <= 15.0:
                break
            
            # Generate a perturbed starting point
            if start_idx == 0:
                x0 = initial_log_roots.copy()
            else:
                rng = np.random.RandomState(start_idx * 42 + 7)
                scale = 0.3 * start_idx
                x0 = initial_log_roots + rng.randn(12) * scale
                # Ensure minimum gap in log-space (roughly)
                x0 = np.sort(x0)
                for i in range(1, 12):
                    if x0[i] < x0[i-1] + 1e-4:
                        x0[i] = x0[i-1] + 1e-4
            
            try:
                if method == 'L-BFGS-B':
                    res = minimize(_log_objective, x0, method=method,
                                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                elif method == 'BFGS':
                    res = minimize(_log_objective, x0, jac=_log_objective_grad,
                                   method=method,
                                   options={'maxiter': 300, 'gtol': 1e-10})
                else:  # Nelder-Mead
                    res = minimize(_log_objective, x0, method=method,
                                   options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                
                if res.success or res.fun < best_val:
                    new_roots = np.exp(res.x)
                    new_roots = np.sort(new_roots)
                    # Enforce minimum gap
                    for i in range(1, 12):
                        if new_roots[i] < new_roots[i-1] + 1e-8:
                            new_roots[i] = new_roots[i-1] + 1e-8
                    
                    if _verify(new_roots):
                        v = _objective(new_roots)
                        if v < best_val:
                            best_roots = new_roots
                            best_val = v
            except Exception:
                continue
        
        # After each method, try a final polish with L-BFGS-B from the best found
        if _remaining() <= 20.0:
            break
        try:
            x0 = np.log(best_roots)
            res = minimize(_log_objective, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < best_val:
                new_roots = np.exp(res.x)
                new_roots = np.sort(new_roots)
                for i in range(1, 12):
                    if new_roots[i] < new_roots[i-1] + 1e-8:
                        new_roots[i] = new_roots[i-1] + 1e-8
                if _verify(new_roots):
                    v = _objective(new_roots)
                    if v < best_val:
                        best_roots = new_roots
                        best_val = v
        except Exception:
            pass
    
    # Final verification
    if not _verify(best_roots):
        best_roots = initial_roots.copy()
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
