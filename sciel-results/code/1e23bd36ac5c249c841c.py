import numpy as np
from scipy.optimize import minimize, differential_evolution
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

    def compute_norm_sq_from_coeffs(c):
        """
        Compute the squared norm of the polynomial given Laguerre coefficients.
        ||P||^2 = sum c_k^2 * (k+1) for alpha=1
        """
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        return norm_sq

    def eval_poly_and_deriv(c, x):
        """
        Evaluate polynomial P(x) = sum c_k L_k^{(1)}(x) and its derivative P'(x) at x.
        Returns (P(x), P'(x)).
        """
        P = 0.0
        dP = 0.0
        for k in range(13):
            if c[k] != 0:
                P += c[k] * eval_genlaguerre(k, 1, x)
                dP += c[k] * eval_genlaguerre(k, 1, x, 1)
        return P, dP

    def penalty_from_coeffs(c, sample_points):
        """
        Compute penalty for not having 12 double roots.
        We check that P(x_i) ≈ 0 and P'(x_i) ≈ 0 at 12 sample points.
        """
        penalty = 0.0
        for x_i in sample_points:
            P_val, dP_val = eval_poly_and_deriv(c, x_i)
            penalty += P_val**2 + dP_val**2
        return penalty

    def cost_func(coeffs):
        """
        Minimize 1/norm_sq + penalty * weight.
        coeffs: 13 Laguerre coefficients.
        """
        norm_sq = compute_norm_sq_from_coeffs(coeffs)
        
        if norm_sq <= 1e-10:
            return 1e10
        
        # Use a fixed set of sample points for penalty
        # These should be in the range where roots might be
        sample_points = np.linspace(0.1, 50.0, 12)
        penalty = penalty_from_coeffs(coeffs, sample_points)
        
        # Weight the penalty heavily to enforce double roots
        weight = 100.0
        return 1.0 / norm_sq + weight * penalty

    # Initial guess: start with the seed program's roots, convert to coefficients
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    coeffs_desc = np.poly(seed_roots)
    p = coeffs_desc[::-1]  # ascending
    c_init = A_T_inv @ p
    
    # Normalize to have reasonable magnitude
    norm_init = np.sqrt(compute_norm_sq_from_coeffs(c_init))
    c_init = c_init / norm_init
    
    # Bounds for coefficients
    bounds = [(-10, 10)] * 13
    
    best_c = c_init.copy()
    best_val = cost_func(c_init)
    
    # Try multiple optimization strategies
    seeds = [42, 123, 456, 789, 101, 202]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        try:
            # First try L-BFGS-B from initial point
            result = minimize(
                cost_func, c_init, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
                
        except Exception:
            pass
        
        # Try differential evolution for global search
        try:
            de_result = differential_evolution(
                cost_func,
                bounds,
                seed=seed,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='latinhypercube'
            )
            
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
                
        except Exception:
            pass

    # Now extract roots from the best coefficients
    # Convert Laguerre coefficients to standard polynomial coefficients
    p = A @ best_c  # p[j] = coeff of x^j in ascending order
    
    # Find roots of the polynomial
    try:
        roots = np.roots(p[::-1])  # np.roots expects descending order
        # Filter for positive real roots
        real_roots = roots[np.abs(roots.imag) < 1e-6].real
        positive_roots = real_roots[real_roots > 1e-6]
        
        if len(positive_roots) >= 12:
            # Take the 12 smallest positive roots (or some selection)
            positive_roots = np.sort(positive_roots)[:12]
            return list(positive_roots)
        elif len(positive_roots) > 0:
            # Pad with reasonable values if we don't have enough
            while len(positive_roots) < 12:
                positive_roots = np.append(positive_roots, positive_roots[-1] * 2.0)
            return list(positive_roots[:12])
    except Exception:
        pass
    
    # Fallback to seed roots
    return [4.0 * (2.0 ** i) for i in range(12)]
