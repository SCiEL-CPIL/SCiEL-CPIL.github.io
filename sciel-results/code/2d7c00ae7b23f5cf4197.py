import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1  # alpha = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
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
            
        # Sort roots to break symmetry and help local optimizers
        roots_sorted = np.sort(roots)
        
        norm_sq = compute_norm_sq(roots_sorted)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search using Differential Evolution in log-space.
    # 2. Local polish using L-BFGS-B.
    
    best_val = 1e10
    best_log_roots = None
    best_roots = None

    # Define bounds for log(roots).
    # Roots are positive. Let's try a wide range.
    # The seed used geometric progressions up to 4*2^11 ~ 8192.
    # Log(8192) ~ 9. Log(0.01) ~ -4.6.
    bounds = [(-5.0, 12.0)] * 12

    # Run Differential Evolution
    # tol is relative tolerance for convergence
    # maxiter is max number of iterations
    # popsize is population size multiplier
    try:
        de_result = differential_evolution(
            cost_func,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            polish=False,  # We will polish manually for better control
            init='latinhypercube'
        )
        
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_log_roots = de_result.x
            best_roots = np.exp(de_result.x)
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE didn't find anything or timed out, fall back to a few local starts
    if best_log_roots is None:
        # Fallback to seed-like starts
        seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
        best_log_roots = seed_log_roots
        best_val = cost_func(seed_log_roots)
        best_roots = np.exp(best_log_roots)

    # Final polish with L-BFGS-B
    if best_log_roots is not None and time.time() - start_time < deadline - 5:
        try:
            polish_result = minimize(
                cost_func, 
                best_log_roots, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x
                best_roots = np.exp(polish_result.x)
        except Exception:
            pass

    # Ensure roots are positive and sorted
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        best_roots = np.sort(best_roots)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
