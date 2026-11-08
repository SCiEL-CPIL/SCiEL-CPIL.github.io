import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p (monomial coefficients, ascending). c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared L2 norm with weight e^{-x} of P(x) = prod(x - r_i).
        """
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1) for alpha=1
        norm_sq = np.sum(c**2 * (np.arange(m + 1) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound = ||P||^2 / (product of roots)
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        prod_roots = np.prod(roots)
        
        if norm_sq <= 1e-10 or prod_roots <= 1e-10:
            return 1e10
            
        return norm_sq / prod_roots

    # Optimization Strategy:
    # 1. Try multiple formulations and initial guesses
    # 2. Use differential evolution with multiple restarts
    # 3. Polish with L-BFGS-B
    
    # Seed: geometric sequence
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-3.0, 8.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Evaluate seed
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    if seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots.copy()
        best_log_roots = seed_log_roots.copy()

    # Run Differential Evolution with multiple restarts
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 90:  # Leave 90s for polish and fallback
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=150, 
                popsize=15, 
                tol=1e-10, 
                polish=True,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 20 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
