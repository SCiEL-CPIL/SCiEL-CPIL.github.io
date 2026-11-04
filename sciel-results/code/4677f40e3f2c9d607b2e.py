"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from math import lgamma, exp
import time
import math

def _compute_c4(roots):
    """
    Compute the C4 upper bound for the Cohn-Goncalves Laguerre construction.
    P(x) = L_1^{(24)}(x) * prod_{i=1}^{12} (x - r_i)^2
    L_1^{(24)}(x) = -x + 25
    C4 = 1 / sqrt( ||P||^2_{L2(e^{-x})} )
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.diff(np.sort(roots)) <= 0):
        return 1e10
    
    # Build Q(x) = prod_i (x - r_i)^2 as polynomial coefficients
    q_coeffs = np.array([1.0])
    for r in roots:
        new_coeffs = np.zeros(len(q_coeffs) + 2)
        for i in range(len(q_coeffs)):
            new_coeffs[i] += q_coeffs[i]
            new_coeffs[i+1] += -2.0 * r * q_coeffs[i]
            new_coeffs[i+2] += r*r * q_coeffs[i]
        q_coeffs = new_coeffs
    
    # P(x) = (25 - x) * Q(x), degree 25
    n_q = len(q_coeffs)
    p_coeffs = np.zeros(n_q + 1)
    for i in range(n_q):
        p_coeffs[i+1] += -q_coeffs[i]
        p_coeffs[i] += 25.0 * q_coeffs[i]
    
    # c[k] = coefficient of x^k in P
    c = p_coeffs[::-1]
    
    # ||P||^2 = sum_{k,j} c_k * c_j * Gamma(k+j+1)
    max_deg = len(c) - 1
    norm_sq = 0.0
    for k in range(max_deg + 1):
        ck = c[k]
        if abs(ck) < 1e-15:
            continue
        for j in range(k, max_deg + 1):
            cj = c[j]
            if abs(cj) < 1e-15:
                continue
            val = ck * cj * exp(lgamma(k + j + 1))
            if k != j:
                val *= 2.0
            norm_sq += val
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e10
    
    return 1.0 / np.sqrt(norm_sq)

def _objective(params):
    """Minimize C4 using log-root parameterization."""
    log_roots = params
    roots = np.exp(log_roots)
    
    if np.any(roots <= 0) or np.any(roots > 1e8) or np.any(roots < 1e-8):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    return _compute_c4(roots)

def construct():
    """Construct 12 positive double-root locations minimizing C4."""
    start_time = time.time()
    deadline = start_time + 570.0
    
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_c4 = _compute_c4(seed_roots)
    
    best_c4 = seed_c4
    best_roots = seed_roots.copy()
    
    # Phase 1: Global search with Differential Evolution
    # This explores the 12-dimensional space more effectively than local methods
    log_roots_bounds = [(-2.0, 10.0)] * 12
    
    # Run a few DE iterations to find a good starting point
    for de_seed in [42, 123]:
        if time.time() > deadline - 300:
            break
        try:
            de_result = differential_evolution(
                _objective,
                log_roots_bounds,
                seed=de_seed,
                maxiter=50,
                popsize=20,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_c4:
                best_c4 = de_result.fun
                best_roots = np.exp(de_result.x)
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B from multiple starting points
    # Use the best DE result and some structured initial points
    
    np.random.seed(42)
    
    initial_points = [np.log(best_roots)]
    
    # Add geometric sequence
    q_geo = (48.0 / 4.0) ** (1.0 / 11.0)
    geo_roots = 4.0 * q_geo ** np.arange(12)
    initial_points.append(np.log(geo_roots))
    
    # Add perturbed seed
    for _ in range(10):
        pert = np.random.normal(0, 0.3, size=12)
        log_roots_try = np.log(seed_roots) + pert
        log_roots_try = np.sort(log_roots_try)
        diffs = np.diff(log_roots_try)
        if np.any(diffs < 0.01):
            log_roots_try = np.cumsum(np.maximum(diffs, 0.01)) + log_roots_try[0]
        initial_points.append(log_roots_try)
    
    for idx, x0 in enumerate(initial_points):
        if time.time() > deadline - 120:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=log_roots_bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_c4:
                best_c4 = result.fun
                best_roots = np.exp(result.x)
                
        except Exception:
            continue
    
    # Phase 3: Final polish with Nelder-Mead if time permits
    if time.time() < deadline - 30:
        try:
            result2 = minimize(
                _objective,
                np.log(best_roots),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-14, 'xatol': 1e-14}
            )
            if result2.fun < best_c4:
                best_c4 = result2.fun
                best_roots = np.exp(result2.x)
        except Exception:
            pass
    
    # Final validation
    if np.any(best_roots <= 0) or np.any(np.diff(best_roots) <= 0):
        return list(seed_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
