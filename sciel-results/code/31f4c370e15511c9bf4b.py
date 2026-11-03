"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre
import time

def _compute_norm_log(roots, n_quad=200):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)
    with weight e^{-x}.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound (assuming Bound ~ 1/Norm).
    """
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log P(x)^2 = 2 * sum log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 2.0 * np.log(diff)
    
    log_integrand = -x_quad + log_p_sq
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf
    
    sum_val = np.sum(w_quad * np.exp(log_integrand - max_log))
    if sum_val <= 0:
        return -np.inf
        
    log_norm_sq = max_log + np.log(sum_val)
    return log_norm_sq

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Candidate initializations:
    # 1. Seed roots: 4, 8, ..., 48
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    # 2. Zeros of Laguerre polynomial L_12
    lag_roots, _ = roots_laguerre(12)
    lag_roots = np.sort(lag_roots)
    
    candidates = [seed_roots.copy(), lag_roots.copy()]
    
    best_roots = None
    best_log_norm = -np.inf
    
    for cand in candidates:
        ln = _compute_norm_log(cand)
        if np.isfinite(ln) and ln > best_log_norm:
            best_log_norm = ln
            best_roots = cand.copy()
    
    if best_roots is None:
        best_roots = seed_roots.copy()
        best_log_norm = _compute_norm_log(best_roots)
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return 1e10
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10
        log_norm = _compute_norm_log(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    # Local refinement from each starting point
    for start in [seed_roots, lag_roots]:
        if time.time() - start_time > deadline - 30:
            break
        try:
            result = minimize(objective, start, method='L-BFGS-B',
                             bounds=[(1e-6, None)] * 12,
                             options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12})
            if np.isfinite(result.fun) and -result.fun > best_log_norm:
                best_log_norm = -result.fun
                best_roots = np.sort(result.x)
        except Exception:
            pass
    
    # Global search with differential evolution if time permits
    if time.time() - start_time < deadline - 120:
        try:
            def de_objective(params):
                elapsed = time.time() - start_time
                if elapsed > deadline - 10:
                    return 1e10
                roots = np.sort(params)
                if any(r <= 1e-6 for r in roots):
                    return 1e10
                log_norm = _compute_norm_log(roots)
                if not np.isfinite(log_norm):
                    return 1e10
                return -log_norm
            
            bounds_de = [(0.1, 60.0)] * 12
            result_de = differential_evolution(
                de_objective, bounds_de, seed=42, maxiter=50,
                popsize=15, tol=1e-8, polish=True,
                options={'maxiter': 50, 'init': 'sobol'}
            )
            if np.isfinite(result_de.fun) and -result_de.fun > best_log_norm:
                best_log_norm = -result_de.fun
                best_roots = np.sort(result_de.x)
        except Exception:
            pass
    
    # Final Nelder-Mead polish if time permits
    if time.time() - start_time < deadline - 60:
        try:
            result2 = minimize(objective, best_roots, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if np.isfinite(result2.fun) and -result2.fun > best_log_norm:
                best_log_norm = -result2.fun
                best_roots = np.sort(result2.x)
        except Exception:
            pass
    
    # Final verification
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = seed_roots.copy()
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
