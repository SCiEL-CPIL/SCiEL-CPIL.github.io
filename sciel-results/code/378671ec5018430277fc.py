import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

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

    def compute_norm_sq_from_squares(squares):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        squares: array of 6 positive numbers (squared root locations).
        The 12 double roots are: sqrt(squares[0]), sqrt(squares[0]), ..., sqrt(squares[5]), sqrt(squares[5])
        """
        # Build the 12 double roots
        roots = np.repeat(np.sqrt(squares), 2)
        
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i) for i in 0..11
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

    def cost_func(log_squares):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        log_squares: array of 6 log(squared root locations).
        """
        squares = np.exp(log_squares)
        
        # Check for numerical issues
        if np.any(squares <= 0) or np.any(np.isnan(squares)) or np.any(np.isinf(squares)):
            return 1e10
            
        norm_sq = compute_norm_sq_from_squares(squares)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 6 log-squared-root parameters using Differential Evolution.
    # 2. Multiple restarts to avoid local minima.
    # 3. Basin hopping for additional exploration.
    # 4. Final polish with L-BFGS-B.
    
    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i for i in 0..11
    # The 12 roots are: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # With double roots, we have 6 unique values: 4, 16, 64, 256, 1024, 4096
    # squares = [16, 256, 4096, 65536, 1048576, 16777216]
    # log_squares = log(16) + i * log(16) for i in 0..5
    seed_log_squares = np.log(16.0) + np.arange(6) * np.log(16.0)
    
    # Bounds for log_squares
    # squares = exp(log_squares), roots = sqrt(squares) = exp(log_squares/2)
    # If roots range from ~0.01 to ~10000, log_roots range from ~-4.6 to ~9.2
    # log_squares = 2 * log_roots, so range is ~-9.2 to ~18.4
    # Let's allow a wider range: [-10, 20]
    bounds = [(-10.0, 20.0)] * 6
    
    best_roots = None
    best_val = 1e10
    best_log_squares = None

    # Run Differential Evolution with multiple restarts
    # We have ~590s. DE is stochastic.
    # Let's run a few independent DEs with different seeds and settings.
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s for polish and fallback
            break
        
        try:
            # Use smaller popsize and maxiter for speed, relying on multiple restarts
            # popsize=15, maxiter=100 is usually fast for 6D.
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=300, 
                popsize=25, 
                tol=1e-10, 
                polish=True,  # L-BFGS-B polish
                init='sobol' # Better initial distribution
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_squares = result.x
                best_roots = np.repeat(np.sqrt(np.exp(best_log_squares)), 2)
                
        except Exception as e:
            pass

    # Basin hopping for additional exploration
    if time.time() - start_time < deadline - 30 and best_log_squares is not None:
        try:
            # Use basin hopping with a local minimizer
            def local_minimizer(x):
                res = minimize(cost_func, x, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 100, 'ftol': 1e-12})
                return res.fun, res.x, {}
            
            bh_result = basinhopping(
                cost_func, 
                best_log_squares, 
                minimizer=local_minimizer,
                niter=30,
                T=1.0,
                stepsize=0.5,
                seed=42
            )
            
            if bh_result.fun < best_val:
                best_val = bh_result.fun
                best_log_squares = bh_result.x
                best_roots = np.repeat(np.sqrt(np.exp(best_log_squares)), 2)
                
        except Exception as e:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    # The seed from the prompt: r0=4, q=2
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_squares)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_squares = seed_log_squares

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 10 and best_log_squares is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_squares, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_squares = polish_result.x
                best_roots = np.repeat(np.sqrt(np.exp(best_log_squares)), 2)
        except Exception:
            pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]

