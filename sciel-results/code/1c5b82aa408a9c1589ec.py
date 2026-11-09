import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    # The polynomial is P(x) = prod_{i=1}^m (x - r_i)^2
    # Degree of P is 2*m = 24.
    # We need Laguerre basis up to degree 24.
    max_deg = 2 * m
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial P(x) = prod (x - r_i)^2 
        in the Laguerre basis L_k^{(1)}.
        roots: array of m positive numbers.
        """
        # Build polynomial Q(x) = prod (x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_Q_desc = np.poly(roots)
        
        # P(x) = Q(x)^2
        # Convolve coeffs_Q_desc with itself to get P_coeffs_desc
        coeffs_P_desc = np.convolve(coeffs_Q_desc, coeffs_Q_desc)
        
        # Convert to ascending order for the matrix multiplication
        p = coeffs_P_desc[::-1]
        
        # Ensure p has length max_deg + 1
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        else:
            p = p[:max_deg + 1]
            
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(max_deg + 1) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Use Differential Evolution for global search.
    # 2. Use L-BFGS-B for local polishing from DE results and geometric seeds.
    
    # Bounds for log_roots
    bounds = [(-4.0, 12.0)] * m
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # 1. Global Search with Differential Evolution
    # We will run DE if time permits.
    # DE parameters:
    # popsize: 20 (20 * 12 = 240 initial points)
    # maxiter: 50
    # This is ~1200 function evaluations. Should be fast.
    
    de_result = None
    try:
        # Use a fixed seed for reproducibility if needed, but random is fine
        # strategy='best1bin' is standard
        de_result = differential_evolution(
            cost_func, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8,
            seed=42,
            polish=False, # We will do our own polish
            init='latinhypercube'
        )
        if de_result.success or de_result.fun < best_val:
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_log_roots = de_result.x
                best_roots = np.exp(best_log_roots)
    except Exception as e:
        print(f"DE failed: {e}")

    # 2. Local Optimization from multiple seeds
    # Seeds:
    # a) The DE result (if available)
    # b) Geometric progressions
    
    seeds_log_roots = []
    
    if de_result is not None and de_result.x is not None:
        seeds_log_roots.append(de_result.x)
    
    # Geometric seeds
    a_values = [1.0, 2.0, 4.0, 8.0, 16.0]
    q_values = [1.5, 1.8, 2.0, 2.2, 2.5]
    
    for a in a_values:
        for q in q_values:
            log_a = np.log(a)
            log_q = np.log(q)
            log_roots = log_a + np.arange(m) * log_q
            seeds_log_roots.append(log_roots)
            
    # Also add the original seed explicitly if not covered
    orig_seed = np.log(4.0) + np.arange(m) * np.log(2.0)
    seeds_log_roots.append(orig_seed)

    # Run local optimization from each seed
    for log_roots_seed in seeds_log_roots:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # Leave 30s for final polish
            break
        
        try:
            result = minimize(
                cost_func, 
                log_roots_seed, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback / Initial guess if all optimizations fail
    seed_roots = 4.0 * (2.0 ** np.arange(m))
    seed_val = cost_func(np.log(seed_roots))
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = np.log(seed_roots)

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
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
    return [4.0 * (2.0 ** i) for i in range(m)]
