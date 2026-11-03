"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature (includes e^{-x} weight)
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - x_i)^2
    # Compute log_P for numerical stability
    log_P = np.zeros(n_quad)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _compute_c4_bound_fast(roots):
    """Faster version using vectorized operations where possible."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
    # Shape: (150, 12)
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
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [x0.copy()]
    
    # Perturbed Laguerre roots
    for scale in [0.1, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Log-uniform spacing (geometric in linear space) - key for concentration
    for log_start in [-2.0, -1.5, -1.0, -0.5, 0.0]:
        for log_end in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 100:
                    initial_points.append(roots.copy())
    
    # Pure uniform spacing
    for start in [0.1, 0.5, 1.0, 2.0]:
        for end in [5.0, 10.0, 20.0, 50.0]:
            if end > start:
                roots = np.linspace(start, end, 12)
                initial_points.append(roots.copy())
    
    # Random log-uniform
    for _ in range(8):
        log_roots = rng.uniform(-2.0, 3.0, size=12)
        log_roots = np.sort(log_roots)
        roots = np.exp(log_roots)
        initial_points.append(roots.copy())
    
    # Structured: geometric with various ratios
    for start in [0.3, 0.5, 1.0]:
        for ratio in [1.3, 1.5, 1.8, 2.0]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 100 and geom[0] > 1e-3:
                initial_points.append(geom.copy())
    
    # Try L-BFGS-B from each initial point
    bounds = [(1e-6, None)] * 12
    
    for x_init in initial_points:
        if time.time() > deadline - 90:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
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
    
    # Multiple DE restarts with different seeds and bounds
    de_configs = [
        {'bounds': [(0.01, 50)] * 12, 'maxiter': 40, 'popsize': 15, 'seed': 42},
        {'bounds': [(0.05, 30)] * 12, 'maxiter': 40, 'popsize': 15, 'seed': 123},
        {'bounds': [(0.1, 20)] * 12, 'maxiter': 35, 'popsize': 18, 'seed': 777},
        {'bounds': [(0.01, 80)] * 12, 'maxiter': 30, 'popsize': 20, 'seed': 999},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 60:
            break
        try:
            de_bounds = cfg['bounds']
            
            result_de = differential_evolution(
                _objective,
                bounds=de_bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                tol=1e-10,
                seed=cfg['seed'],
                polish=False,
                updating='immediate',
                workers=1
            )
            
            if result_de.fun < best_score:
                candidate = np.sort(result_de.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
            
            # Polish the DE result with tight tolerances
            if np.all(result_de.x > 0):
                result_polish = minimize(
                    _objective,
                    result_de.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if np.all(result_polish.x > 0) and not np.any(np.isnan(result_polish.x)):
                    candidate = np.sort(result_polish.x)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
                        
        except Exception:
            pass
    
    # Final high-precision polish on best roots
    if time.time() < deadline - 10:
        try:
            # Multiple polish passes with progressively tighter tolerances
            for ftol, gtol, maxit in [(1e-14, 1e-12, 300), (1e-15, 1e-13, 300), (1e-16, 1e-14, 200)]:
                if time.time() > deadline - 5:
                    break
                result_final = minimize(
                    _objective,
                    best_roots,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxit, 'ftol': ftol, 'gtol': gtol}
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
# EVOLVE-BLOCK-END
