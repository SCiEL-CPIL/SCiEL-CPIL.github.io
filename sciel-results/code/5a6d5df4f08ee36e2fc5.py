"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound_fast(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    Uses Gauss-Laguerre quadrature for numerical integration.
    Metric: int(P^2 * w) / (int(P * w))^2
    where P(x) = prod (x - r_i)^2 and w(x) = e^{-x}
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature (includes e^{-x} weight)
    # 200 points provides high accuracy for this smooth integrand
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2
    # Compute log_P for numerical stability
    # Vectorized: diff has shape (n_quad, 12)
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    
    # Avoid log(0) if a root coincides with a quadrature node (unlikely but possible)
    # Add a small epsilon
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Strategy:
    1. Use standard Laguerre roots as a strong initial guess.
    2. Perform global optimization using Differential Evolution in log-space.
    3. Refine with local optimization (L-BFGS-B) in log-space.
    4. Return the best set of roots found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Initial guess: Zeros of Laguerre L_12(x)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    # If we are already close to the deadline, return immediately
    if time.time() > deadline - 60:
        return [float(xi) for xi in best_roots]
    
    # Objective function for log-space optimization
    # Parameters are log(roots), so roots = exp(params)
    def objective_log(log_roots):
        roots = np.exp(log_roots)
        # Ensure roots are sorted for consistency (though product is commutative)
        # Sorting is not strictly necessary for the integral, but good for determinism
        # roots = np.sort(roots)
        return _compute_c4_bound_fast(roots)
    
    # Initial guess in log-space
    log_x0 = np.log(x0)
    
    # Phase 1: Global Optimization with Differential Evolution
    # Bounds for log(roots):
    # Min root ~ 1e-3 -> log ~ -6.9
    # Max root ~ 100 -> log ~ 4.6
    # We can widen these slightly to allow exploration
    bounds_log = [(-8.0, 6.0)] * 12
    
    try:
        # Use differential evolution for global search
        # maxiter=100, popsize=15 is a reasonable balance for 12D
        # We restrict time by checking deadline
        de_result = differential_evolution(
            objective_log,
            bounds_log,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-8,
            atol=1e-8,
            init='sobol', # Better initialization
            update_interval=1
        )
        
        if de_result.success or de_result.fun < best_score:
            candidate_roots = np.exp(de_result.x)
            candidate_roots = np.sort(candidate_roots)
            score = _compute_c4_bound_fast(candidate_roots)
            if score < best_score:
                best_score = score
                best_roots = candidate_roots.copy()
                log_best_roots = de_result.x.copy()
            else:
                log_best_roots = np.log(best_roots)
        else:
            log_best_roots = np.log(best_roots)
            
    except Exception:
        log_best_roots = np.log(best_roots)
    
    # Phase 2: Local Refinement with L-BFGS-B in log-space
    # This helps fine-tune the roots for higher precision
    if time.time() < deadline - 30:
        try:
            local_result = minimize(
                objective_log,
                log_best_roots,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if local_result.fun < best_score:
                candidate_roots = np.exp(local_result.x)
                candidate_roots = np.sort(candidate_roots)
                score = _compute_c4_bound_fast(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots.copy()
        except Exception:
            pass
    
    # Phase 3: Final High-Precision Polish
    # Sometimes a second local pass helps if the first one converged to a plateau
    if time.time() < deadline - 10:
        try:
            log_current = np.log(best_roots)
            polish_result = minimize(
                objective_log,
                log_current,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if polish_result.fun < best_score:
                candidate_roots = np.exp(polish_result.x)
                candidate_roots = np.sort(candidate_roots)
                score = _compute_c4_bound_fast(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots.copy()
        except Exception:
            pass
    
    # Ensure the output is a list of 12 positive floats
    result = [max(float(r), 1e-6) for r in best_roots]
    
    # Final validation
    if len(result) != 12 or any(r <= 0 for r in result):
        # Fallback to Laguerre roots if something went wrong
        result = [float(xi) for xi in x0]
        
    return result
# EVOLVE-BLOCK-END
