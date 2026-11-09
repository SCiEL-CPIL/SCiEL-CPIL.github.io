import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s limit

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # New structural approach: Use the fact that we can parameterize via
    # elementary symmetric polynomials or via a different root structure.
    # Try optimizing in terms of log-spaced roots with a variable base and ratio.
    
    def cost_param(log_base, log_ratio):
        """
        Parameterize roots as r_i = base * ratio^i for i=0..11
        This reduces 12D to 2D but may miss optimal solutions.
        """
        base = np.exp(log_base)
        ratio = np.exp(log_ratio)
        
        if base <= 0 or ratio <= 0 or np.isnan(base) or np.isnan(ratio):
            return 1e10
            
        roots = base * (ratio ** np.arange(12))
        
        if np.any(roots <= 0) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Phase 1: Quick 2D search over (log_base, log_ratio)
    elapsed = time.time() - start_time
    if elapsed < deadline - 120:
        try:
            bounds_2d = [(-2.0, 8.0), (-1.0, 2.0)]  # log_base, log_ratio
            result_2d = differential_evolution(
                cost_param, 
                bounds_2d, 
                seed=42, 
                maxiter=100, 
                popsize=15, 
                tol=1e-10, 
                polish=True,
                init='sobol'
            )
            if result_2d.fun < best_val:
                best_val = result_2d.fun
                base = np.exp(result_2d.x[0])
                ratio = np.exp(result_2d.x[1])
                best_roots = base * (ratio ** np.arange(12))
                best_log_roots = np.log(best_roots)
        except Exception:
            pass

    # Phase 2: Full 12D optimization from multiple starting points
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        try:
            # Use smaller popsize and maxiter for speed
            result = differential_evolution(
                cost_func, 
                [(-3.0, 9.0)] * 12, 
                seed=seed, 
                maxiter=150, 
                popsize=15, 
                tol=1e-8, 
                polish=False,  # Disable polish for speed, we'll do it later
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_log_roots = np.log(seed_roots)
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=[(-3.0, 9.0)] * 12, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
