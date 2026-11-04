import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
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

def _objective_log(log_roots):
    """Objective in log-space for better constraint handling."""
    roots = np.exp(log_roots)
    return _compute_c4_bound_fast(roots)

def _get_equilibrium_roots():
    """Get roots from the corrected mean-field equilibrium condition."""
    # For the Dyson gas with linear confining potential V'(x)=1,
    # the equilibrium condition is sum_{j!=i} 2/(r_i - r_j) = 1
    # We solve this numerically
    
    m = 12
    # Start from Laguerre roots as initial guess
    x0, _ = roots_laguerre(m)
    
    # Use fixed-point iteration to solve the equilibrium condition
    roots = x0.copy()
    for iteration in range(1000):
        new_roots = roots.copy()
        for i in range(m):
            # sum_{j!=i} 2/(r_i - r_j) = 1
            s = 0.0
            for j in range(m):
                if j != i:
                    s += 2.0 / (roots[i] - roots[j])
            # Newton step: if s != 1, adjust root
            # The derivative of s w.r.t. r_i is -2 * sum_{j!=i} 1/(r_i - r_j)^2
            ds = 0.0
            for j in range(m):
                if j != i:
                    ds -= 2.0 / (roots[i] - roots[j])**2
            if abs(ds) > 1e-15:
                new_roots[i] = roots[i] - (s - 1.0) / ds
        
        if np.max(np.abs(new_roots - roots)) < 1e-12:
            roots = new_roots
            break
        roots = new_roots
    
    # Ensure all roots are positive
    roots = np.maximum(roots, 1e-6)
    roots = np.sort(roots)
    return roots

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Get initial guesses from different strategies
    laguerre_roots, _ = roots_laguerre(12)
    laguerre_roots = np.sort(laguerre_roots)
    
    equilibrium_roots = _get_equilibrium_roots()
    
    best_roots = equilibrium_roots.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    # Also try Laguerre roots
    lag_score = _compute_c4_bound_fast(laguerre_roots)
    if lag_score < best_score:
        best_score = lag_score
        best_roots = laguerre_roots.copy()
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [equilibrium_roots.copy(), laguerre_roots.copy()]
    
    # Perturbed versions
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        perturbed = best_roots * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Scaled versions
    for scale in [0.8, 1.2, 1.5, 2.0]:
        scaled = best_roots * scale
        initial_points.append(scaled.copy())
    
    # Random initial points
    for _ in range(5):
        rand_roots = rng.uniform(0.1, 50, size=12)
        rand_roots = np.sort(rand_roots)
        initial_points.append(rand_roots)
    
    # Optimize in log-space for better constraint handling
    bounds_log = [(-20, 20)] * 12  # log(1e-9) to log(1e9)
    
    for x_init in initial_points:
        if time.time() > deadline - 60:
            break
        
        try:
            x_init_log = np.log(np.maximum(x_init, 1e-10))
            
            result = minimize(
                _objective_log,
                x_init_log,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success or np.isfinite(result.fun):
                candidate = np.exp(result.x)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final polish in log-space with tighter tolerance
    if time.time() < deadline - 10:
        try:
            best_roots_log = np.log(best_roots)
            result_final = minimize(
                _objective_log,
                best_roots_log,
                method='L-BFGS-B',
                bounds=bounds_log,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.isfinite(result_final.fun):
                candidate = np.exp(result_final.x)
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
