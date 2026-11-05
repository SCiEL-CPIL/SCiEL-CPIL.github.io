"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre, roots_genlaguerre
from scipy.stats import norm

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
    
    # Precompute weights for norm
    weights = np.arange(13) + 1  # weights[k] = k+1

    def compute_p_and_c(roots):
        """
        Compute polynomial coefficients p (ascending) and Laguerre coefficients c.
        roots: array of 12 positive numbers.
        Returns: (p, c) where p is ascending coeffs, c is Laguerre coeffs.
        """
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        c = A_T_inv @ p
        return p, c

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        p, c = compute_p_and_c(roots)
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def compute_gradient(log_roots):
        """
        Compute gradient of cost = 1/norm_sq with respect to log_roots.
        
        Let r_i = exp(log_roots[i]), P(x) = prod(x - r_i).
        dP/dr_i = P(x) / (x - r_i)
        dP/d(log_r_i) = r_i * dP/dr_i = r_i * P(x) / (x - r_i)
        
        We need d(norm_sq)/d(log_r_i) where norm_sq = c^T W c, W = diag(1,2,...,13)
        c = A_T_inv @ p
        dc/d(log_r_i) = A_T_inv @ dp/d(log_r_i)
        
        dp/d(log_r_i) = r_i * dP/dr_i in ascending basis.
        """
        roots = np.exp(log_roots)
        p, c = compute_p_and_c(roots)
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return np.zeros(12)
        
        # Gradient of norm_sq wrt log_roots
        grad_norm_sq = np.zeros(12)
        
        for i in range(12):
            # dP/dr_i = P(x) / (x - r_i)
            # In coefficient form: if P(x) = sum p_j x^j, then
            # dP/dr_i = sum_{j=0}^{11} p_j x^j / (x - r_i)
            # This is a rational function. We can compute it via polynomial division.
            
            # Alternative: use the fact that P(x) = (x - r_i) * Q_i(x)
            # where Q_i(x) = prod_{j != i}(x - r_j)
            # Then dP/dr_i = Q_i(x)
            
            # Compute Q_i by dividing P by (x - r_i)
            # Use synthetic division
            r_i = roots[i]
            
            # P in descending: coeffs_desc
            coeffs_desc = np.poly(roots)
            # Divide by (x - r_i): quotient has degree 11
            # Synthetic division
            q_coeffs = np.zeros(12)  # descending, degree 11
            q_coeffs[0] = coeffs_desc[0]  # leading coeff is 1
            for j in range(1, 12):
                q_coeffs[j] = coeffs_desc[j] + r_i * q_coeffs[j-1]
            
            # q_coeffs is descending for Q_i(x)
            # Convert to ascending
            q_asc = q_coeffs[::-1]
            
            # dp/d(log_r_i) = r_i * Q_i in ascending basis
            dp_dlogr_i = r_i * q_asc
            
            # dc/d(log_r_i) = A_T_inv @ dp_dlogr_i
            dc_dlogr_i = A_T_inv @ dp_dlogr_i
            
            # d(norm_sq)/d(log_r_i) = 2 * c^T W dc_dlogr_i
            grad_norm_sq[i] = 2.0 * np.sum(c * weights * dc_dlogr_i)
        
        # d(cost)/d(log_roots) = d(1/norm_sq)/d(log_roots) = -grad_norm_sq / norm_sq^2
        grad_cost = -grad_norm_sq / (norm_sq ** 2)
        
        return grad_cost

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

    def cost_func_with_grad(log_roots):
        """Cost function with gradient for L-BFGS-B."""
        val = cost_func(log_roots)
        if val >= 1e9:
            return val, np.zeros(12)
        grad = compute_gradient(log_roots)
        return val, grad

    def cost_func_lognormal(log_mu, log_sigma):
        """
        Cost function for log-normal parameterization.
        roots_i = exp(mu + sigma * z_i) where z_i are standard normal quantiles
        for i = 0..11.
        """
        # Quantiles of standard normal for 12 points
        # Use equal spacing in probability space: (i+0.5)/12
        probs = (np.arange(12) + 0.5) / 12.0
        # Inverse CDF of standard normal
        z = norm.ppf(probs)
        
        mu = math.exp(log_mu) if log_mu is not None else log_mu
        sigma = math.exp(log_sigma) if log_sigma is not None else log_sigma
        
        # Ensure sigma is positive
        if sigma <= 0:
            return 1e10
            
        roots = mu * np.exp(sigma * z)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Optimization Strategy:
    # 1. Global search over 12 log-root parameters using Differential Evolution (fewer restarts).
    # 2. Diverse initial points from structured families.
    # 3. High-precision polish with L-BFGS-B using analytical gradients.
    # 4. Final Newton-style refinement.
    
    # Seed: geometric sequence a=4, q=2
    # r_i = 4 * 2^i
    # log_r_i = log(4) + i * log(2)
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None
    
    # Run Differential Evolution with fewer restarts (each is expensive)
    seeds = [42, 123]
    
    for seed in seeds:
        if not time_ok():
            break
        
        try:
            # Use smaller popsize and maxiter for speed
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=80, 
                popsize=12, 
                tol=1e-8, 
                polish=False,  # We'll do our own polish
                init='sobol'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception as e:
            pass

    # Fallback / Initial guess if DE fails or to ensure we have a baseline
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_log_roots = seed_log_roots.copy()
        best_roots = seed_roots.copy()

    # Generate diverse initial points from structured families
    initial_points = [best_log_roots.copy()]
    
    # Geometric sequences with different parameters
    for a in [1.0, 2.0, 3.0, 5.0, 8.0]:
        for q in [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
            if not time_ok():
                break
            roots_try = a * (q ** np.arange(12))
            if np.all(roots_try > 0) and np.all(roots_try < 1e10):
                log_roots_try = np.log(roots_try)
                # Clip to bounds
                for i in range(12):
                    log_roots_try[i] = max(bounds[i][0], min(bounds[i][1], log_roots_try[i]))
                initial_points.append(log_roots_try)
    
    # Random perturbations around best
    for pseed in [100, 200, 300, 400, 500]:
        if not time_ok():
            break
        rng = np.random.RandomState(pseed)
        noise = rng.normal(0, 0.5, size=12)
        x0 = best_log_roots + noise
        for i in range(12):
            x0[i] = max(bounds[i][0], min(bounds[i][1], x0[i]))
        initial_points.append(x0)

    # NEW: Add initial points from Laguerre polynomial roots (Jacobi matrix strategy)
    # The roots of L_m^(alpha)(x) are positive and well-distributed.
    # We use them as seeds for the optimization.
    if time_ok():
        for alpha in [0.0, 1.0, 2.0, 0.5, 1.5]:
            try:
                # Get roots of L_12^(alpha)(x)
                # roots_genlaguerre(n, alpha) returns roots of L_n^(alpha)
                lag_roots = roots_genlaguerre(12, alpha).real
                # Filter positive roots (they should all be positive for alpha > -1)
                lag_roots = lag_roots[lag_roots > 0]
                if len(lag_roots) == 12:
                    log_lag_roots = np.log(lag_roots)
                    # Clip to bounds
                    for i in range(12):
                        log_lag_roots[i] = max(bounds[i][0], min(bounds[i][1], log_lag_roots[i]))
                    initial_points.append(log_lag_roots)
            except Exception:
                pass

    # High-precision polish with L-BFGS-B using analytical gradients
    if time_ok():
        try:
            # Polish each initial point
            for idx, x0 in enumerate(initial_points):
                if not time_ok():
                    break
                
                # Run L-BFGS-B with tight tolerance
                res = minimize(
                    cost_func_with_grad,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
                )
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(best_log_roots)
                    
        except Exception as e:
            pass
    
    # Final high-precision Newton-style refinement
    if best_log_roots is not None and time_ok():
        try:
            # Compute Hessian approximation using finite differences for a few steps
            x = best_log_roots.copy()
            for newton_step in range(20):
                if not time_ok():
                    break
                
                # Compute gradient
                f, g = cost_func_with_grad(x)
                
                # If gradient is small enough, stop
                if np.linalg.norm(g) < 1e-14:
                    break
                
                # Estimate Hessian using forward differences
                h = 1e-6
                H = np.zeros((12, 12))
                for i in range(12):
                    x_plus = x.copy()
                    x_plus[i] += h
                    _, g_plus = cost_func_with_grad(x_plus)
                    H[i, :] = (g_plus - g) / h
                    H[:, i] = H[i, :]  # Symmetrize
                H = (H + H.T) / 2.0
                
                # Solve H dx = -g
                try:
                    dx = np.linalg.solve(H + 1e-12 * np.eye(12), -g)
                except np.linalg.LinAlgError:
                    dx = -0.01 * g  # Fallback to steepest descent
                
                # Line search to ensure decrease
                alpha = 1.0
                for _ in range(30):
                    x_new = x + alpha * dx
                    # Clip to bounds
                    for i in range(12):
                        x_new[i] = max(bounds[i][0], min(bounds[i][1], x_new[i]))
                    f_new = cost_func(x_new)
                    if f_new < f:
                        break
                    alpha *= 0.5
                
                x = x_new
                f = f_new
                if f < best_val:
                    best_val = f
                    best_log_roots = x.copy()
                    best_roots = np.exp(x)
                    
        except Exception as e:
            pass

    # Ensure roots are positive and sorted (though order doesn't matter for the product)
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
