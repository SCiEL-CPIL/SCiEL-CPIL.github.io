"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
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

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def _objective_from_log(log_roots):
    """Objective in log-space to ensure positivity."""
    roots = np.exp(log_roots)
    return _compute_c4_bound_fast(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Precompute quadrature points for speed
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Start from standard Laguerre roots as initial guess
    from scipy.special import roots_laguerre
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 60:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(12345)
    
    # Strategy 1: Geometric parameterization in log-space
    # roots_i = exp(a + i*b) for i=0..11, so we optimize over (a, b)
    # This enforces sorted, positive roots with a smooth 2D landscape
    
    def _geometric_objective(params):
        a, b = params
        # Ensure roots are positive and spread out
        # a = log(r0), b = log(q) where q > 1
        if b <= 0:
            return 1e10
        roots = np.exp(a + b * np.arange(12))
        # Ensure all roots are positive and not too extreme
        if np.any(roots <= 0) or np.any(roots > 1e6) or np.any(roots < 1e-6):
            return 1e10
        return _compute_c4_bound_fast(roots)
    
    # Multistart over geometric parameterization
    for a_init in np.linspace(-2, 2, 20):
        for b_init in np.linspace(0.1, 1.0, 20):
            if time.time() > deadline - 30:
                break
            try:
                result = minimize(
                    _geometric_objective,
                    np.array([a_init, b_init]),
                    method='L-BFGS-B',
                    bounds=[(-5, 5), (0.05, 2.0)],
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if result.fun < best_score:
                    a, b = result.x
                    roots = np.exp(a + b * np.arange(12))
                    if np.all(roots > 0) and np.all(roots < 1e6):
                        candidate = np.sort(roots)
                        score = _compute_c4_bound_fast(candidate)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate.copy()
            except Exception:
                pass
        if time.time() > deadline - 30:
            break
    
    # Strategy 2: Log-space optimization in 12D
    # roots_i = exp(log_roots_i), optimize over log_roots
    
    def _log_objective(log_roots):
        roots = np.exp(log_roots)
        if np.any(roots <= 0) or np.any(np.isnan(roots)):
            return 1e10
        return _compute_c4_bound_fast(roots)
    
    # Initialize log_roots from Laguerre roots
    log_x0 = np.log(x0 + 1e-10)
    
    # Multistart in log-space
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        
        if trial == 0:
            log_init = log_x0.copy()
        else:
            # Perturb in log-space
            log_init = log_x0 + rng.standard_normal(12) * 0.5
        
        try:
            result = minimize(
                _log_objective,
                log_init,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * 12,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if result.fun < best_score:
                candidate = np.exp(result.x)
                if np.all(candidate > 0) and np.all(candidate < 1e6):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    # Strategy 3: Direct optimization with sorting constraint handled by symmetry
    # Use the best roots found so far as starting points
    
    if time.time() < deadline - 20:
        try:
            result_final = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * 12,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result_final.fun < best_score:
                candidate = np.sort(result_final.x)
                if np.all(candidate > 0):
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
