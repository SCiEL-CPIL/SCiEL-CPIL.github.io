"""Refined uncertainty inequality - Cohn-Goncalves Laguerre construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_laguerre

# Precompute Gauss-Laguerre quadrature nodes and weights (fixed for performance)
_N_QUAD = 300
_X_NODES, _X_WEIGHTS = np.polynomial.laguerre.laggauss(_N_QUAD)
_LOG_WEIGHTS = np.log(_X_WEIGHTS + 1e-300)

def _compute_bound(roots):
    """
    Compute an upper bound related to the Cohn-Goncalves Laguerre construction.
    roots: array of 12 positive double-root locations.
    Returns a scalar bound (lower is better).
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    
    # Sort roots
    roots = np.sort(roots)
    
    # Check for reasonable spacing (no overlaps)
    if np.any(np.diff(roots) < 1e-8):
        return 1e10
    
    # Compute the Laguerre-weighted L2 norm of P(x) = prod (x - r_i)^2
    # Integral I = ∫_0^∞ |P(x)|^2 e^{-x} dx = ∫_0^∞ [prod (x-r_i)^4] e^{-x} dx
    # Using Gauss-Laguerre quadrature: I ≈ sum w_i * [prod (x_i - r_j)^4]
    
    # log|P(x)|^2 = 4 * sum_j log|x - r_j|
    # We compute log_terms = log(w_i) + 4 * sum_j log|x_i - r_j|
    
    log_abs_P_sq = np.zeros_like(_X_NODES)
    for r in roots:
        # Avoid log(0)
        log_abs_P_sq += 4.0 * np.log(np.abs(_X_NODES - r) + 1e-300)
    
    log_terms = log_abs_P_sq + _LOG_WEIGHTS
    max_log = np.max(log_terms)
    
    # If the value is astronomically large, return a large penalty
    if max_log > 700:
        return np.exp(700)
    
    # Compute sum of exp(log_terms - max_log) * exp(max_log)
    # To avoid underflow in exp(log_terms - max_log) for very negative terms,
    # we rely on the fact that if max_log is not huge, the subtraction is safe.
    # If max_log is very negative, exp(log_terms - max_log) is fine.
    integral = np.exp(max_log) * np.sum(np.exp(log_terms - max_log))
    
    return integral

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin from 630s limit
    
    # The seed returns [4, 8, 12, ..., 48] and scores 1.0
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Pre-compute seed bound for normalization
    seed_bound = _compute_bound(seed_roots)
    
    best_roots = seed_roots.copy()
    best_bound = 1.0  # Normalized
    
    def objective(x):
        if time.time() > deadline - 10:
            return best_bound
        if np.any(x <= 0):
            return 1e10
        raw = _compute_bound(x)
        if raw >= 1e9:
            return 1e10
        return raw / seed_bound
    
    # Try differential evolution for global optimization
    if time.time() < deadline - 120:
        try:
            bounds = [(0.1, 80.0)] * 12
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=50,
                tol=1e-8,
                seed=42,
                polish=True,
                update_interval=1,
                callback=lambda xk, convergence: (time.time() < deadline - 30)
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Refine with L-BFGS-B from the best found
    if time.time() < deadline - 60:
        try:
            result = minimize(
                objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(0.01, 100)] * 12,
                options={'maxiter': 500, 'maxfun': 10000}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Try a few more local optimizations from different starts
    additional_starts = [
        np.linspace(2, 40, 12),
        np.linspace(0.5, 25, 12),
        np.sort(np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], dtype=float)),
    ]
    
    for x0 in additional_starts:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0.01, 100)] * 12,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            result = minimize(
                objective,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass
    
    # Ensure roots are sorted and positive
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0):
        best_roots = seed_roots.copy()
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
