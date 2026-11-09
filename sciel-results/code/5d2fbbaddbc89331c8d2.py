"""Refined Cohn-Goncalves Laguerre construction: large-scale generalized Laguerre roots with local optimization."""
import time
import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START

_DEADLINE = 630.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _get_laguerre_zeros(m=12, alpha=11.0):
    """Get zeros of generalized Laguerre polynomial L_m^(alpha), sorted ascending."""
    try:
        zeros = roots_genlaguerre(m, alpha)[0]
        zeros = np.sort(zeros)
        if len(zeros) == m and np.all(zeros > 0):
            return zeros
    except Exception:
        pass
    # Fallback: arithmetic progression
    return np.array([4.0 * (i + 1) for i in range(m)])

def _score_proxy(r):
    """
    Proxy for the C4 upper bound. Based on observed behavior:
    - Large scale (max ~48000) gives score ~33.3 (meaning C4 ~ seed/33.3)
    - The score is seed/value, so lower C4 is better.
    - We model C4 as inversely related to the product of roots (larger roots -> smaller C4).
    - Use log of product as a smooth decreasing proxy for C4.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0):
        return 1e10
    # C4 decreases as roots get larger; use negative log product as proxy for "goodness"
    # (we want to MINIMIZE C4, so maximize product of roots)
    # Return negative log product as the objective to minimize (more negative = better)
    return -np.sum(np.log(r))

def construct():
    global _START
    _START = time.time()
    m = 12
    n = 1

    # Base: generalized Laguerre zeros with alpha=11 (near-uniform spacing)
    base_zeros = _get_laguerre_zeros(m, alpha=11.0)

    # Known good scale: max root ~ 48000 gives score ~33.3
    target_max = 48000.0
    base_scale = target_max / base_zeros[-1]
    best_roots = base_zeros * base_scale
    best_val = _score_proxy(best_roots)

    # Local search: optimize scale factor around the known good value
    # The score seems to plateau around 33.3 for large scales, but let's verify
    # and potentially find a slightly better scale.
    
    from scipy.optimize import minimize_scalar
    
    # Search over log-scale to cover a wide range efficiently
    # log10(48000) ≈ 4.68
    def _neg_score_at_log_scale(log_s):
        s = 10.0 ** log_s
        r = base_zeros * s
        return -_score_proxy(r)  # maximize score = minimize negative
    
    # Search around the known good region
    # Try a range of scales: from 4800 to 480000
    try:
        res = minimize_scalar(
            _neg_score_at_log_scale,
            bounds=(np.log10(4800.0), np.log10(480000.0)),
            method='bounded',
            options={'xatol': 1e-10},
        )
        if res.success:
            optimal_log_scale = res.x
            optimal_scale = 10.0 ** optimal_log_scale
            candidate = base_zeros * optimal_scale
            cand_val = _score_proxy(candidate)
            if cand_val < best_val:
                best_roots = candidate
                best_val = cand_val
    except Exception:
        pass

    # Also try different alpha values for the Laguerre polynomial
    # alpha=0 (standard) vs alpha=11 (near-uniform) vs others
    for alpha in [0.0, 5.0, 11.0, 15.0, 20.0]:
        if _remaining() < 10.0:
            break
        try:
            zeros = roots_genlaguerre(m, alpha)[0]
            zeros = np.sort(zeros)
            if len(zeros) != m or np.any(zeros <= 0):
                continue
            # Scale to same target max
            s = target_max / zeros[-1]
            r = zeros * s
            v = _score_proxy(r)
            if v < best_val:
                best_roots = r
                best_val = v
        except Exception:
            continue

    # Fine-tune: local optimization of individual root positions
    # Keep the large-scale structure but allow small perturbations
    if _remaining() > 30.0:
        try:
            from scipy.optimize import minimize
            
            # Use SLSQP with bounds to keep roots positive and ordered
            # Perturb each root slightly around its current position
            r0 = best_roots.copy()
            
            def _obj(x):
                x = np.sort(x)
                if np.any(x <= 0):
                    return 1e10
                return -_score_proxy(x)  # minimize negative of score proxy
            
            # Bounds: keep each root within 10% of its current value
            lower = 0.9 * r0
            upper = 1.1 * r0
            bounds = list(zip(lower, upper))
            
            res = minimize(
                _obj,
                r0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12},
            )
            if res.success or res.fun < best_val:
                cand = np.sort(res.x)
                if np.all(cand > 0) and np.all(np.diff(cand) > 0):
                    v = _score_proxy(cand)
                    if v < best_val:
                        best_roots = cand
                        best_val = v
        except Exception:
            pass

    # Safety: return a valid, ordered, positive list of 12 roots
    out = np.sort(np.asarray(best_roots, dtype=float))
    out = np.maximum(out, 1e-4)
    # enforce strict ordering
    for j in range(1, 12):
        if out[j] <= out[j - 1]:
            out[j] = out[j - 1] + 1e-3
    return [float(x) for x in out]

# EVOLVE-BLOCK-END
