import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580

    m = 12
    # We assume P(x) = Q(x)^2 where Q is degree 6 with 6 positive roots.
    # This ensures P has 6 double roots (total 12 roots counting multiplicity).
    
    # Precompute transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)
    
    d = np.arange(m + 1) + 1

    try:
        A_inv = np.linalg.inv(A)
        K = A_inv @ np.diag(d) @ A_inv.T
    except np.linalg.LinAlgError:
        A_pinv = np.linalg.pinv(A)
        K = A_pinv @ np.diag(d) @ A_pinv.T

    def compute_norm_sq_from_q_roots(q_roots):
        """
        Compute norm squared of P(x) = Q(x)^2 where Q(x) = prod(x - r_i).
        q_roots: array of 6 positive numbers.
        """
        # Build Q(x) coefficients (descending)
        # Q(x) = x^6 + c5 x^5 + ... + c0
        q_coeffs_desc = np.poly(q_roots)
        
        # P(x) = Q(x)^2
        # Use convolution to square the polynomial
        p_coeffs_desc = np.convolve(q_coeffs_desc, q_coeffs_desc)
        
        # p_coeffs_desc is length 13 (degree 12)
        # Convert to ascending for K matrix
        p = p_coeffs_desc[::-1]
        
        # Compute norm squared
        norm_sq = float(p @ K @ p)
        
        if norm_sq < 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_q_roots):
        """
        Minimize 1/norm_sq.
        log_q_roots: array of 6 log-roots.
        """
        q_roots = np.exp(log_q_roots)
        if np.any(q_roots <= 0) or np.any(np.isnan(q_roots)) or np.any(np.isinf(q_roots)):
            return 1e10
            
        norm_sq = compute_norm_sq_from_q_roots(q_roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy
    # Search space: 6 variables (log of the 6 roots of Q)
    # Seed: Geometric sequence for Q roots?
    # If P roots were 4, 8, ..., 8192 (12 roots), Q roots would be sqrt of those?
    # No, if P = Q^2, and P has roots r_1..r_12, then Q has roots r_1..r_6 (duplicated).
    # So if we want P to have "double roots", the 6 roots of Q are the locations.
    # Let's start with a geometric sequence for Q roots.
    # e.g., 2, 4, 8, 16, 32, 64
    
    seed_q_roots = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    seed_log_q_roots = np.log(seed_q_roots)
    
    bounds = [(-4.0, 10.0)] * 6

    best_val = 1e10
    best_log_q_roots = seed_log_q_roots.copy()
    best_q_roots = seed_q_roots.copy()
    
    seed_val = cost_func(seed_log_q_roots)
    best_val = seed_val

    # Add other seeds
    # Uniform spacing
    r_lin = np.linspace(1.0, 100.0, 6)
    seeds = [seed_log_q_roots, np.log(r_lin)]
    
    # Try roots from the 12-root solution mapped to 6?
    # If we had 12 roots, we could pair them. But we don't have a good 12-root solution yet.
    
    for seed_log in seeds:
        if time.time() > deadline - 100:
            break
        try:
            res = minimize(
                cost_func, seed_log, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_q_roots = np.exp(res.x)
                best_log_q_roots = res.x.copy()
        except Exception:
            pass

    # Global search with Differential Evolution on 6D space
    de_runs = 0
    remaining_time = deadline - time.time()
    max_de_runs = 5
    for i in range(max_de_runs):
        if time.time() > deadline - 120:
            break
        de_seed = 123 + i * 100
        try:
            result = differential_evolution(
                cost_func, bounds, seed=de_seed, maxiter=200,
                popsize=20, tol=1e-10, polish=True, init='sobol'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_q_roots = np.exp(result.x)
                best_log_q_roots = result.x.copy()
            de_runs += 1
        except Exception:
            pass

    # Final polish
    if best_log_q_roots is not None and time.time() < deadline - 10:
        try:
            polish_result = minimize(
                cost_func, best_log_q_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_q_roots = np.exp(polish_result.x)
                best_log_q_roots = polish_result.x.copy()
        except Exception:
            pass

    # Construct the 12 roots for P(x)
    # P(x) has double roots at the locations of Q(x)'s roots.
    # So the 12 roots are r_1, r_1, r_2, r_2, ..., r_6, r_6.
    final_roots = np.repeat(best_q_roots, 2)
    
    # Sort them just in case, though order doesn't matter for the polynomial
    final_roots = np.sort(final_roots)
    
    if final_roots is not None:
        final_roots = np.maximum(final_roots, 1e-6)
        return list(final_roots)

    # Fallback
    return [4.0 * (2.0 ** i) for i in range(12)]

