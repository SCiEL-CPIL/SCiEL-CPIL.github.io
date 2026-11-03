import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 150
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

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Strategy: Use the asymptotic equilibrium of Laguerre zeros as an initial guess,
    # then refine with local optimization. The Laguerre zeros themselves are the
    # stationary points of the potential, so they should be a very good starting point.
    
    # Get the standard Laguerre roots of degree 12 (these are the equilibrium points)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points around the Laguerre roots
    initial_points = [x0.copy()]
    
    # Perturbed Laguerre roots with different scales
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Add some structured variations
    # Scaled Laguerre roots
    for scale in [0.8, 1.2, 1.5, 2.0]:
        scaled = x0 * scale
        initial_points.append(scaled.copy())
    
    # Random uniform in [0.1, 30]
    for _ in range(3):
        rand_roots = rng.uniform(0.1, 30, size=12)
        rand_roots = np.sort(rand_roots)
        initial_points.append(rand_roots)
    
    # Geometric spacing variations
    for start in [0.5, 1.0, 2.0]:
        for ratio in [1.3, 1.5, 1.8]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 100 and geom[0] > 0.1:
                initial_points.append(geom.copy())
    
    # Try L-BFGS-B from each initial point
    bounds = [(1e-6, None)] * 12
    
    for x_init in initial_points:
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            candidate = result.x
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final polish on best roots with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
