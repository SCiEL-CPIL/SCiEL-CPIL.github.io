"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution, minimize, fsolve

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute transformation matrix from standard basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)

    # c = A_T_inv @ p where p is standard-basis coeffs (ascending)
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute squared Laguerre norm of P(x) = prod(x - r_i).
        """
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # Convert to Laguerre basis
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize upper bound on C4 ~ 1/||P||^2.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def cost_with_grad(log_roots):
        """
        Cost and analytic gradient using polynomial division.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros_like(log_roots)
        
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # Convert to Laguerre basis
        c = A_T_inv @ p
        
        # Norm squared
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros_like(log_roots)
        
        cost = 1.0 / norm_sq
        
        # Compute gradient analytically
        grad = np.zeros(12)
        
        for i in range(12):
            other_roots = np.delete(roots, i)
            coeffs_other_desc = np.poly(other_roots)
            p_other = coeffs_other_desc[::-1]  # ascending
            
            dp_dr_i = -p_other
            dc_dr_i = A_T_inv @ dp_dr_i
            dnorm_sq_dr_i = 2.0 * np.sum(c * (np.arange(13) + 1) * dc_dr_i)
            dcost_dr_i = -dnorm_sq_dr_i / (norm_sq * norm_sq)
            grad[i] = dcost_dr_i * roots[i]
        
        return cost, grad

    def equilibrium_residuals(roots):
        """
        Residuals for the mean-field equilibrium condition:
        sum_{j != i} 2/(r_i - r_j) + 1/r_i = 1
        """
        res = np.zeros(12)
        for i in range(12):
            s = 0.0
            for j in range(12):
                if i != j:
                    s += 2.0 / (roots[i] - roots[j])
            res[i] = s + 1.0 / roots[i] - 1.0
        return res

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Strategy 1: Solve equilibrium equations to get a high-quality initial guess
    # Use the geometric sequence as a starting point for fsolve
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    
    try:
        # fsolve requires a good initial guess. 
        # The geometric sequence might be far. 
        # Let's try to solve it.
        sol_eq, info, ier, msg = fsolve(equilibrium_residuals, seed_roots, full_output=True)
        if ier == 1 and np.all(sol_eq > 0):
            # Check if this solution is better
            log_sol = np.log(sol_eq)
            val_eq = cost_func(log_sol)
            if val_eq < best_val:
                best_val = val_eq
                best_roots = sol_eq.copy()
                best_log_roots = log_sol.copy()
    except Exception:
        pass

    # Strategy 2: Direct optimization from geometric seed
    seed_log_roots = np.log(seed_roots)
    val_seed = cost_func(seed_log_roots)
    if val_seed < best_val:
        best_val = val_seed
        best_roots = seed_roots.copy()
        best_log_roots = seed_log_roots.copy()

    # Strategy 3: Polish with L-BFGS-B
    if best_log_roots is not None and time_ok():
        # Define bounds for log roots to keep them positive and reasonable
        # Roots should be in a range that doesn't overflow or underflow
        bounds = [(-2.0, 10.0)] * 12
        
        try:
            polish_result = minimize(
                cost_with_grad,
                best_log_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 50}
            )
            
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Strategy 4: If time permits, try a few random restarts with L-BFGS-B
    if time_ok() and (time.time() - start_time) < (DEADLINE - 60.0):
        np.random.seed(42)
        for _ in range(5):
            if not time_ok():
                break
            # Random log-roots in a reasonable range
            rand_log_roots = np.random.uniform(-1.0, 5.0, 12)
            try:
                res = minimize(
                    cost_with_grad,
                    rand_log_roots,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-2.0, 10.0)] * 12,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_log_roots = res.x.copy()
                    best_roots = np.exp(best_log_roots)
            except Exception:
                pass

    # Final polish if time permits
    if best_log_roots is not None and time_ok():
        try:
            polish_result2 = minimize(
                cost_with_grad,
                best_log_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-2.0, 10.0)] * 12,
                options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100}
            )
            if polish_result2.fun < best_val:
                best_val = polish_result2.fun
                best_log_roots = polish_result2.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END
