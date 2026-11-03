"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from scipy.special import eval_genlaguerre, roots_genlaguerre

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound from the Laguerre polynomial construction.
    P(t) = prod_i (t - x_i)^2 * L_m^(n)(t)
    C4 ~ 1 / ||P||_{L2(w)} where w(t) = t^n * exp(-t)
    """
    x = np.asarray(roots, dtype=np.float64)
    if x.shape != (m,) or np.any(x <= 0) or np.any(np.isnan(x)):
        return 1e10
    
    # Use Gauss-Laguerre quadrature with sufficient points
    # Degree of P is 3m. 4*3m + 10 is usually enough, let's use 500 for safety.
    n_quad = 500
    tq, wq = roots_genlaguerre(n_quad, n)
    
    # Evaluate Laguerre polynomial at quadrature points
    L_vals = eval_genlaguerre(m, n, tq)
    
    # Evaluate the product prod_i (t - x_i)^2 at quadrature points
    # Vectorized computation: (tq[None, :] - x[:, None])
    # Shape: (m, n_quad)
    diffs = tq[None, :] - x[:, None]
    # To avoid log(0) if a root coincides with a quadrature node (unlikely but possible),
    # we add a tiny epsilon or just handle it.
    # Actually, log(0) is -inf, exp(-inf) is 0. This is mathematically correct (P(tq_j)=0).
    # However, numerical stability might suffer if diffs are very small.
    # Let's compute product of squares directly in log space.
    log_prod_sq = np.sum(2.0 * np.log(np.abs(diffs)), axis=0)
    
    prod_sq = np.exp(log_prod_sq)
    
    P_vals = prod_sq * L_vals
    
    # L2 norm squared under Laguerre weight
    norm_sq = np.sum(wq * P_vals ** 2)
    
    if norm_sq <= 0 or np.isnan(norm_sq) or not np.isfinite(norm_sq):
        return 1e10
    
    return 1.0 / np.sqrt(norm_sq)

def _objective_log_roots(log_roots, m=12, n=1):
    """Objective in log-space: minimize C4 bound given log-roots."""
    if np.any(np.isnan(log_roots)) or np.any(np.isinf(log_roots)):
        return 1e10
    log_roots_clipped = np.clip(log_roots, -10, 30)
    roots = np.exp(log_roots_clipped)
    roots = np.sort(roots)
    if np.any(roots <= 0) or np.any(roots > 1e12):
        return 1e10
    return _compute_c4_bound(roots, m, n)

def construct():
    m = 12
    n = 1
    
    start_time = time.time()
    deadline = 600.0
    
    best_score = 1e10
    best_roots = None
    
    # Strategy:
    # 1. Coarse grid search over geometric progression parameters (x0, r)
    # 2. Local refinement with L-BFGS-B on free roots
    # 3. Final local refinement
    
    # Step 1: Geometric Grid Search
    # Roots x_i = x0 * r^(i-1) for i=1..m
    # We want to cover a reasonable range of x0 and r.
    # x0 can range from 0.1 to 10.
    # r can range from 0.8 to 1.5.
    
    x0_vals = np.linspace(0.5, 10.0, 20)
    r_vals = np.linspace(0.9, 1.3, 20)
    
    best_geo_score = 1e10
    best_geo_roots = None
    
    for x0 in x0_vals:
        for r in r_vals:
            if time.time() - start_time > deadline - 300:
                break
            roots = x0 * (r ** np.arange(m))
            if np.any(roots <= 0) or np.any(roots > 1e6):
                continue
            score = _compute_c4_bound(roots, m, n)
            if score < best_geo_score:
                best_geo_score = score
                best_geo_roots = roots.copy()
        
        if time.time() - start_time > deadline - 300:
            break
    
    if best_geo_roots is None:
        # Fallback to a reasonable default if grid search fails or times out
        best_geo_roots = np.array([(i + 1) ** 1.5 for i in range(m)])
        best_geo_roots = np.sort(best_geo_roots)
        best_geo_score = _compute_c4_bound(best_geo_roots, m, n)
    
    best_roots = best_geo_roots
    best_score = best_geo_score
    
    # Step 2: Local Refinement on Free Roots
    # Start from the best geometric configuration
    if time.time() - start_time < deadline - 60:
        log_roots_init = np.log(best_roots)
        try:
            res = minimize(_objective_log_roots, log_roots_init, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13,
                                   'maxls': 100})
            if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                best_score = res.fun
                best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
        except Exception:
            pass
    
    # Step 3: Second Refinement (Nelder-Mead or L-BFGS-B again)
    if time.time() - start_time < deadline - 30:
        log_roots_init = np.log(best_roots)
        try:
            res = minimize(_objective_log_roots, log_roots_init, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            if res.fun < best_score and res.fun < 1e9 and np.all(np.isfinite(res.x)):
                best_score = res.fun
                best_roots = np.sort(np.exp(np.clip(res.x, -10, 30)))
        except Exception:
            pass
            
    # Final check and cleanup
    if best_roots is None:
        best_roots = np.array([(i + 1) ** 1.5 for i in range(m)])
        best_roots = np.sort(best_roots)
    
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
