"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 600.0  # seconds wall-clock limit
    SAFETY_MARGIN = 30.0  # return 30s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            # L_k^{(1)}(x) = sum_{j=0}^k (-1)^j * binom(k+1, k-j) * x^j / j!
            # Note: binom(k+1, k-j) = binom(k+1, j+1)
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # We want to solve for c in: p = A^T @ c  =>  c = (A^T)^{-1} @ p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    # Precompute weights for the norm calculation: w_k = k+1
    weights = np.arange(13) + 1.0

    def compute_cost(log_roots):
        """
        Compute the cost (upper bound on C4) given log-roots.
        Cost = 1 / ||P||^2
        """
        # Convert to actual roots
        # Clip log_roots to prevent overflow/underflow in exp
        # exp(700) is inf, exp(-745) is 0.
        # Our roots are positive, so log_roots can be any real number, 
        # but physically they shouldn't be too extreme.
        # Let's clip to [-50, 50] to be safe against exp overflow.
        lr = np.clip(log_roots, -50.0, 50.0)
        roots = np.exp(lr)
        
        # Check for invalid roots
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e100
            
        # Build polynomial coefficients in standard basis (ascending order)
        # np.poly(roots) returns descending coefficients [1, c1, ..., cm]
        try:
            coeffs_desc = np.poly(roots)
        except Exception:
            return 1e100
            
        p = coeffs_desc[::-1] # Ascending: p[0] is constant, p[12] is x^12 coeff
        
        # Transform to Laguerre basis coefficients
        c = A_T_inv @ p
        
        # Calculate norm squared: sum c_k^2 * (k+1)
        norm_sq = np.dot(c**2, weights)
        
        if norm_sq <= 1e-10 or np.isnan(norm_sq) or np.isinf(norm_sq):
            return 1e100
            
        # The upper bound on C4 is proportional to 1/norm_sq
        # We minimize this value.
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search using Differential Evolution (DE) in 12D log-space.
    # 2. Multiple restarts with different seeds to explore the landscape.
    # 3. Final polish with L-BFGS-B for high precision.
    
    # Bounds for log-roots
    # Seed roots: 4 * 2^i -> log(4) to log(4*2^11) ~ 1.38 to 9.0
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(100000) ~ 11.5
    # Using -10 to 20 in log-space is safe.
    bounds = [(-10.0, 20.0)] * 12
    
    best_val = 1e100
    best_log_roots = None
    best_roots = None

    # Seed initialization for DE: Center around the geometric seed
    # Seed log roots: log(4) + i*log(2)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Run Differential Evolution with multiple seeds
    # DE is stochastic, so we run several instances.
    # Time budget: ~570s. Each DE run takes variable time.
    # We will run DE as long as time_ok() is true.
    
    de_seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in de_seeds:
        if not time_ok():
            break
            
        try:
            # Use 'sobol' initialization for better coverage
            # maxiter and popsize balance speed and quality
            # 12D problem: popsize=20, maxiter=150 is usually fast enough
            result = differential_evolution(
                compute_cost, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                popsize=25, 
                tol=1e-10, 
                polish=True,  # DE includes an L-BFGS-B polish at the end
                init='sobol',
                update_interval=1
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(np.clip(best_log_roots, -50.0, 50.0))
                
        except Exception as e:
            # If DE fails, continue to next seed
            continue

    # Fallback: If DE didn't find anything better, use the seed
    seed_val = compute_cost(seed_log_roots)
    if best_log_roots is None or seed_val < best_val:
        best_val = seed_val
        best_log_roots = seed_log_roots
        best_roots = np.exp(seed_log_roots)

    # Final high-precision polish with L-BFGS-B
    # This ensures we are at a local minimum with high precision.
    if time_ok() and best_log_roots is not None:
        try:
            # Tighter bounds for polish to avoid wandering
            polish_bounds = [(-20.0, 30.0)] * 12
            
            polish_result = minimize(
                compute_cost, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=polish_bounds, 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(np.clip(best_log_roots, -50.0, 50.0))
        except Exception:
            pass

    # Ensure roots are positive and valid
    if best_roots is None:
        # Absolute fallback
        best_roots = 4.0 * (2.0 ** np.arange(12))
    else:
        # Clean up any tiny negatives or NaNs
        best_roots = np.maximum(best_roots, 1e-10)
        
    return list(best_roots)
# EVOLVE-BLOCK-END
