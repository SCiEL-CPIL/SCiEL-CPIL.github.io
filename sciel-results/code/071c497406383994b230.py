import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s kill

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))
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

    # Seed: geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    # Roots should be positive.
    # If roots are too small or too large, the norm might behave badly.
    # Based on seed (4 to 4*2^11 ~ 8192), log range is [~1.4, ~9].
    # Let's allow a wider range: log(0.01) ~ -4.6 to log(10000) ~ 9.2
    bounds = [(-4.0, 10.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Strategy:
    # 1. Run a single long DE with large population for thorough exploration.
    # 2. Collect multiple good candidates from the DE run.
    # 3. Local polish on each candidate.
    
    # Single long DE run with larger population
    try:
        result = differential_evolution(
            cost_func, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=25, 
            tol=1e-10, 
            polish=False,  # We'll do our own polishing
            init='sobol',
            callback=None
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_log_roots = result.x
            best_roots = np.exp(best_log_roots)
            
    except Exception:
        pass

    # Check time - if we've used too much time, skip to fallback
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for polishing, return what we have
        if best_roots is None:
            best_roots = 4.0 * (2.0 ** np.arange(12))
            best_log_roots = seed_log_roots
            best_val = cost_func(seed_log_roots)
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)

    # Multi-start local optimization from multiple starting points
    # Generate diverse starting points
    start_points = []
    
    # Add the DE result
    if best_log_roots is not None:
        start_points.append(best_log_roots.copy())
    
    # Add the seed
    start_points.append(seed_log_roots.copy())
    
    # Add perturbed versions of the best
    if best_log_roots is not None:
        for scale in [0.1, 0.3, 0.5, 1.0]:
            perturbed = best_log_roots + np.random.RandomState(42).normal(0, scale, size=12)
            perturbed = np.clip(perturbed, -4.0, 10.0)
            start_points.append(perturbed)
    
    # Add some random points in the bounds
    rng = np.random.RandomState(123)
    for _ in range(5):
        rand_point = rng.uniform(-4.0, 10.0, size=12)
        start_points.append(rand_point)
    
    # Polish each starting point
    for i, start_point in enumerate(start_points):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        try:
            polish_result = minimize(
                cost_func, start_point, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(polish_result.x)
        except Exception:
            pass

    # Fallback: seed initialization
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]

BET: {mechanism: "Combining the structural robustness of the current program's single long DE run with the wider bounds and multi-start polishing strategy from the inspirations, which allows for better exploration of the root space and more thorough local optimization, potentially escaping local minima that stalled previous approaches.", q: 0.65, delta: [0.5, 2.0]}
