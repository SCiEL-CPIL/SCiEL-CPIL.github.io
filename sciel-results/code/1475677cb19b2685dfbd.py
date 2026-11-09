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

    # Optimization Strategy:
    # 1. Generate diverse warm starts using Laguerre polynomial roots
    # 2. Multi-restart Basin Hopping with L-BFGS-B polishing
    # 3. Final high-precision polish
    
    # Generate warm starts from Laguerre polynomial roots
    def get_laguerre_warm_start(k_offset=0):
        """
        Get roots of L_{12+k_offset}^{(1)}(x) as a warm start.
        These are known to be good initial points for this type of problem.
        """
        k = 12 + k_offset
        # Use scipy to find roots of generalized Laguerre polynomial
        from numpy.polynomial.laguerre import laggauss
        # laggauss gives roots of L_n(x) with weight e^{-x}, not L_n^{(1)}(x)
        # We need to use scipy.special.roots_genlaguerre
        from scipy.special import roots_genlaguerre
        x, w = roots_genlaguerre(k, 1)
        return x

    # Collect warm starts from different Laguerre polynomials
    warm_starts = []
    for k_off in range(0, 8):
        try:
            roots = get_laguerre_warm_start(k_off)
            # Filter to positive roots only and take 12
            pos_roots = roots[roots > 0]
            if len(pos_roots) >= 12:
                warm_starts.append(pos_roots[:12])
        except Exception:
            pass
    
    # Add the seed geometric sequence
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    warm_starts.append(seed_roots)
    
    # Add some random warm starts
    for _ in range(5):
        random_roots = np.exp(np.random.uniform(0.5, 8.0, 12))
        warm_starts.append(random_roots)
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Basin Hopping from each warm start
    for ws_idx, ws_roots in enumerate(warm_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # Leave 30s for final polish
            break
        
        try:
            ws_log_roots = np.log(ws_roots)
            
            # Quick local polish first
            local_res = minimize(
                cost_func, ws_log_roots, method='L-BFGS-B',
                bounds=[(0.1, 15.0)] * 12,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            current_best = local_res.fun
            current_best_log = local_res.x
            
            # Basin hopping from this local minimum
            if time.time() - start_time < deadline - 20:
                def local_minimizer(x):
                    res = minimize(cost_func, x, method='L-BFGS-B', 
                                 bounds=[(0.1, 15.0)] * 12,
                                 options={'maxiter': 100, 'ftol': 1e-12})
                    return res.fun, res.x, {}
                
                bh_result = basinhopping(
                    cost_func, 
                    current_best_log, 
                    minimizer=local_minimizer,
                    niter=15,
                    T=0.8,
                    stepsize=0.4,
                    seed=42 + ws_idx
                )
                
                if bh_result.fun < current_best:
                    current_best = bh_result.fun
                    current_best_log = bh_result.x
            
            if current_best < best_val:
                best_val = current_best
                best_log_roots = current_best_log
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Fallback to seed if nothing better found
    seed_log_roots = np.log(seed_roots)
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Final high-precision polish pass
    if time.time() - start_time < deadline - 10 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=[(0.1, 15.0)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
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
