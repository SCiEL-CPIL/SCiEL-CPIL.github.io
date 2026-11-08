import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^(1)(x) = sum_{j=0}^k A[k,j] x^j
    # For alpha=1: A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # c = A_T_inv @ p where p is monomial coefficients (ascending)
    A_T_inv = np.linalg.inv(A.T)
    
    # Precompute weight vector for norm
    weights = np.arange(13) + 1  # (k+1) for k=0..12

    def compute_norm_sq_fast(roots):
        """Compute squared norm efficiently."""
        # Build polynomial: P(x) = prod(x - r_i)
        # Use np.poly for descending coefficients, then reverse for ascending
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1].astype(float)
        
        # Transform to Laguerre basis
        c = A_T_inv @ p
        
        # Norm squared
        norm_sq = np.dot(c * c, weights)
        
        if norm_sq <= 1e-15:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """Minimize 1/norm_sq (equivalent to maximizing norm_sq)."""
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        
        norm_sq = compute_norm_sq_fast(roots)
        
        if norm_sq <= 1e-15:
            return 1e10
            
        return 1.0 / norm_sq

    # Seed: geometric sequence r_i = 4 * 2^i
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots: allow wide range
    bounds = [(-3.0, 11.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Strategy 1: Local optimization from the seed point
    try:
        result = minimize(
            cost_func, seed_log_roots, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_log_roots = result.x
            best_roots = np.exp(best_log_roots)
    except Exception:
        pass

    # Strategy 2: Try perturbed seeds
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        perturbations = [
            0.1 * np.ones(12),
            0.1 * np.linspace(-1, 1, 12),
            0.2 * np.random.RandomState(42).randn(12),
            0.3 * np.random.RandomState(123).randn(12),
            0.05 * np.arange(12) * np.ones(12) - 0.3,
        ]
        
        for pert in perturbations:
            if time.time() - start_time > deadline - 100:
                break
            try:
                x0 = seed_log_roots + pert
                result = minimize(
                    cost_func, x0, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_log_roots = result.x
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Strategy 3: Differential Evolution with moderate settings
    elapsed = time.time() - start_time
    if elapsed < deadline - 180:
        seeds = [42, 123, 789]
        for seed in seeds:
            if time.time() - start_time > deadline - 150:
                break
            try:
                result = differential_evolution(
                    cost_func, 
                    bounds, 
                    seed=seed, 
                    maxiter=100, 
                    popsize=15, 
                    tol=1e-8, 
                    polish=True,
                    init='sobol'
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_log_roots = result.x
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Fallback to seed if nothing better found
    if best_roots is None:
        best_roots = 4.0 * (2.0 ** np.arange(12))
        best_log_roots = seed_log_roots
        best_val = cost_func(best_log_roots)

    # Final high-precision polish
    if time.time() - start_time < deadline - 20 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-8)
        return list(best_roots)
    
    return [4.0 * (2.0 ** i) for i in range(12)]
