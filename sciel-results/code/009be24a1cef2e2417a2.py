import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s kill

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute the weight vector for norm calculation
    weight_vec = np.arange(m + 1) + 1  # (k+1) for k=0..m

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.dot(c**2, weight_vec)
        
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

    # Informed initial guesses based on Laguerre root asymptotics
    # For large m, zeros of L_m^(1) are approximately 4m + 2*sqrt(2m)*sqrt(1+4j/m)
    # But we want to maximize the norm, so let's try several geometric/arithmetic progressions
    
    def make_initial_guesses():
        guesses = []
        
        # Geometric sequences with various ratios and starts
        for a in [2.0, 4.0, 8.0, 16.0, 32.0]:
            for q in [1.5, 2.0, 2.5, 3.0]:
                roots = a * (q ** np.arange(12))
                if np.max(roots) < 1e6:  # Keep in reasonable range
                    guesses.append(np.log(roots))
        
        # Arithmetic progressions
        for start in [1.0, 4.0, 8.0, 16.0]:
            for step in [1.0, 2.0, 4.0, 8.0]:
                roots = start + step * np.arange(12)
                if np.min(roots) > 0.1:
                    guesses.append(np.log(roots))
        
        # Near the expected Laguerre root region
        # For m=12, alpha=1, roots are roughly in [1, 60]
        for center in [10.0, 20.0, 30.0, 40.0]:
            for spread in [0.5, 1.0, 2.0]:
                roots = center + spread * np.arange(-6, 6)
                if np.min(roots) > 0.1:
                    guesses.append(np.log(roots))
        
        # Original seed
        guesses.append(np.log(4.0) + np.arange(12) * np.log(2.0))
        
        return guesses

    # Tighter bounds based on expected optimal range
    # Roots likely in range [0.5, 100000] -> log range [-0.7, 11.5]
    bounds = [(-0.7, 11.5)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Strategy:
    # 1. Quick local optimization from multiple informed starting points
    # 2. DE for broader exploration
    # 3. Final high-precision polish on best candidate
    
    initial_guesses = make_initial_guesses()
    
    # Phase 1: Local optimization from multiple starts
    for guess in initial_guesses:
        elapsed = time.time() - start_time
        if elapsed > deadline - 120:  # Leave 120s for DE and final polish
            break
        
        try:
            result = minimize(
                cost_func, guess, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Phase 2: Differential Evolution for broader search
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for final polish
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=100, 
                popsize=12, 
                tol=1e-10, 
                polish=False,
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback: seed initialization
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(np.log(seed_roots))
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = np.log(seed_roots)

    # Phase 3: Final high-precision polish pass
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
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
