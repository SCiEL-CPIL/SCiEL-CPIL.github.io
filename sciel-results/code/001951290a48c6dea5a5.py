"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize_scalar, differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 30.0  # return 30s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # We want c such that P(x) = sum c_k L_k^{(1)}(x)
    # P(x) = sum p_j x^j
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # So P(x) = sum_j p_j x^j = sum_k c_k (sum_j A[k,j] x^j) = sum_j (sum_k c_k A[k,j]) x^j
    # Thus p_j = sum_k c_k A[k,j] => p = A^T c => c = (A^T)^{-1} p
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            # L_k^{(alpha)}(x) = sum_{j=0}^k (-1)^j * binom(k+alpha, k-j) * x^j / j!
            # For alpha=1: (-1)^j * binom(k+1, k-j) / j!
            # binom(k+1, k-j) = binom(k+1, j+1)
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. We can precompute the inverse of A.T
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_coefficients(roots):
        """Compute Laguerre coefficients c_k for P(x) = prod(x - r_i)"""
        # Build polynomial coefficients in standard basis
        # p[0] + p[1]x + ... + p[12]x^12
        # Roots are positive, so factors are (x - r_i)
        # np.poly(roots) returns [1, c1, c2, ..., cm] where P(x) = x^m + c1 x^{m-1} + ...
        # We need p in ascending order: p[0] is constant term.
        coeffs_desc = np.poly(roots) # [1, -e1, e2, ..., (-1)^12 e12]
        p = coeffs_desc[::-1] # [(-1)^12 e12, ..., -e1, 1]
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        return c

    def compute_bound(roots):
        """
        Compute the upper bound on C4.
        The bound is related to 1 / ||P||^2_Laguerre.
        We want to minimize the bound, so we want to maximize ||P||^2.
        """
        if any(r <= 0 for r in roots):
            return 1e10
        
        c = compute_coefficients(roots)
        
        # Norm squared in Laguerre basis for n=1:
        # <L_k^{(1)}, L_l^{(1)}> = delta_{kl} * (k+1)! / (1! k!) = delta_{kl} * (k+1)
        norm_sq = sum(c[k]**2 * (k+1) for k in range(13))
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def get_roots_from_alpha(alpha):
        """
        Get roots of L_12^{(alpha)}(x) using companion matrix for stability.
        Returns sorted array of positive real roots, or None if invalid.
        """
        # Compute coefficients of L_m^{(alpha)}(x) in ascending order (constant to x^m)
        # Then convert to descending for np.roots
        coeffs_asc = np.zeros(m + 1)
        for k in range(m + 1):
            # Coeff of x^k in L_m^{(alpha)}(x) is (-1)^k * binom(m+alpha, m-k) / k!
            try:
                # Use log-gamma for numerical stability
                log_binom = (math.lgamma(m + alpha + 1) - math.lgamma(m - k + 1) - math.lgamma(k + alpha + 1))
                binom_val = math.exp(log_binom)
            except (ValueError, OverflowError):
                return None
            coeffs_asc[k] = ((-1)**k) * binom_val / math.factorial(k)
        
        # Convert to descending order for np.roots
        coeffs_desc = coeffs_asc[::-1]
        
        # Use companion matrix for better numerical stability
        try:
            roots = np.roots(coeffs_desc)
        except Exception:
            return None
        
        # Filter for real positive roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-8:
                real_part = r.real
                if real_part > 1e-10:
                    real_roots.append(real_part)
        
        if len(real_roots) != m:
            return None
        
        return np.array(sorted(real_roots))

    def polish_roots(roots, max_iters=50, tol=1e-8):
        """
        Local polish: minimize the bound by adjusting each root slightly.
        Uses Nelder-Mead on log-space to maintain positivity.
        """
        if not time_ok():
            return roots
        
        # Work in log-space to ensure positivity
        log_roots = np.log(roots)
        
        def log_cost(log_r):
            r = np.exp(log_r)
            if any(r <= 0) or any(~np.isfinite(r)):
                return 1e10
            return compute_bound(r)
        
        try:
            res = minimize(log_cost, log_roots, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol})
            if res.success or res.fun < compute_bound(roots):
                new_roots = np.exp(res.x)
                if all(r > 0 for r in new_roots) and all(np.isfinite(r) for r in new_roots):
                    return new_roots
        except Exception:
            pass
        
        return roots

    # Strategy: 1D search over alpha, then local polish
    best_bound = 1e10
    best_roots = None
    best_alpha = None
    
    # Initial coarse scan over a wider range
    # Roots of L_m^{(alpha)} for large alpha are ~ alpha + m/2 ± sqrt(alpha*m)
    # For m=12, to get roots spanning orders of magnitude, alpha needs to vary
    alpha_min = -5.0
    alpha_max = 100.0
    num_points = 300
    
    for i in range(num_points):
        if not time_ok():
            break
        alpha = alpha_min + (alpha_max - alpha_min) * i / (num_points - 1)
        roots = get_roots_from_alpha(alpha)
        if roots is None:
            continue
        bound = compute_bound(roots)
        if bound < best_bound:
            best_bound = bound
            best_roots = roots.copy()
            best_alpha = alpha

    # Refine around the best alpha found
    if best_alpha is not None:
        def alpha_cost(alpha):
            roots = get_roots_from_alpha(alpha)
            if roots is None:
                return 1e10
            return compute_bound(roots)
        
        # Bounded minimization around best
        local_min = max(-5.0, best_alpha - 5.0)
        local_max = min(100.0, best_alpha + 5.0)
        
        try:
            res = minimize_scalar(alpha_cost, bounds=(local_min, local_max), 
                                  method='bounded', options={'xatol': 1e-8})
            if res.success:
                refined_alpha = res.x
                roots = get_roots_from_alpha(refined_alpha)
                if roots is not None:
                    bound = compute_bound(roots)
                    if bound < best_bound:
                        best_bound = bound
                        best_roots = roots.copy()
                        best_alpha = refined_alpha
        except Exception:
            pass

    # Fallback if 1D search failed
    if best_roots is None:
        r0 = 4.0
        q = 2.0
        best_roots = np.array([r0 * (q ** i) for i in range(12)])
        best_bound = compute_bound(best_roots)
    else:
        # Polish the roots found from the best alpha
        if time_ok():
            polished = polish_roots(best_roots)
            if polished is not None:
                polished_bound = compute_bound(polished)
                if polished_bound < best_bound:
                    best_bound = polished_bound
                    best_roots = polished
    
    # Final: ensure all roots are positive and finite
    best_roots = np.array(best_roots, dtype=float)
    best_roots = best_roots[best_roots > 1e-10]
    if len(best_roots) != 12:
        # Fallback to geometric
        r0 = 4.0
        q = 2.0
        best_roots = np.array([r0 * (q ** i) for i in range(12)])
    
    return list(best_roots)
# EVOLVE-BLOCK-END
