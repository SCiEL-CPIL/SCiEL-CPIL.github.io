"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Gauss-Laguerre nodes and weights
N_QUAD = 150
X_QUAD, W_QUAD = np.polynomial.laguerre.laggauss(N_QUAD)

def _compute_c4_from_log_roots(log_roots):
    """Compute C4 bound from log-roots. log_roots is an array of length 12."""
    # Convert to linear roots
    roots = np.exp(log_roots)
    
    # Check for validity (though exp is always > 0, check for inf/nan)
    if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return np.inf
        
    # Vectorized computation of P(x)
    # diff shape: (N_QUAD, 12)
    diff = X_QUAD[:, np.newaxis] - roots[np.newaxis, :]
    
    # Compute log P(x) = 2 * sum(log|x - r_i|)
    # Add small epsilon to avoid log(0)
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    # P(x) = exp(log_P)
    # To avoid overflow/underflow issues in the integral, we can scale P.
    # However, C4 = int(P^2) / int(P)^2 is scale invariant.
    # Let's just compute P and P^2.
    # If log_P is very large, exp(log_P) might overflow.
    # Max log_P approx 2 * 12 * log(max_x) ~ 24 * log(100) ~ 110. exp(110) is ~ 1e47, safe for float64.
    # Min log_P approx -inf if x is close to root.
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(W_QUAD * P)
    int_P2 = np.sum(W_QUAD * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
        
    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective_log(log_roots):
    """Objective function for optimization in log-space."""
    return _compute_c4_from_log_roots(log_roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    log_x0 = np.log(x0)
    
    best_log_roots = log_x0.copy()
    best_score = _objective_log(best_log_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in np.exp(best_log_roots)]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points in log-space
    initial_points = [log_x0.copy()]
    
    # Perturbed Laguerre roots in log-space
    for scale in [0.1, 0.2, 0.3, 0.5]:
        perturbed_log = log_x0 + scale * rng.standard_normal(12)
        initial_points.append(perturbed_log)
    
    # Log-uniform spacing (linear in log-space)
    for log_start in [-2.0, -1.5, -1.0, -0.5, 0.0]:
        for log_end in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                initial_points.append(log_roots)
    
    # Pure uniform spacing in linear space -> log
    for start in [0.1, 0.5, 1.0, 2.0]:
        for end in [5.0, 10.0, 20.0, 50.0]:
            if end > start:
                roots = np.linspace(start, end, 12)
                initial_points.append(np.log(roots))
    
    # Random log-uniform
    for _ in range(8):
        log_roots = rng.uniform(-2.0, 3.0, size=12)
        log_roots = np.sort(log_roots)
        initial_points.append(log_roots)
    
    # Structured: geometric with various ratios (linear in log-space)
    for start_log in [-1.0, -0.5, 0.0]:
        for ratio_log in [0.1, 0.2, 0.3, 0.4]: # log(ratio)
            log_roots = start_log + ratio_log * np.arange(12)
            initial_points.append(log_roots)
    
    # Try L-BFGS-B from each initial point in log-space
    # Bounds for log-roots: roots in [1e-6, 100] -> log in [-13.8, 4.6]
    log_bounds = [(-13.8, 4.6)] * 12
    
    for x_init in initial_points:
        if time.time() > deadline - 90:
            break
        
        try:
            result = minimize(
                _objective_log,
                x_init,
                method='L-BFGS-B',
                bounds=log_bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            candidate_log = result.x
            if not np.any(np.isnan(candidate_log)):
                # Ensure within bounds
                candidate_log = np.clip(candidate_log, -13.8, 4.6)
                score = _objective_log(candidate_log)
                if score < best_score:
                    best_score = score
                    best_log_roots = candidate_log.copy()
        except Exception:
            pass
    
    # Multiple DE restarts with different seeds and bounds in log-space
    de_configs = [
        {'bounds': [(-5.0, 3.0)] * 12, 'maxiter': 40, 'popsize': 15, 'seed': 42},
        {'bounds': [(-4.0, 2.0)] * 12, 'maxiter': 40, 'popsize': 15, 'seed': 123},
        {'bounds': [(-3.0, 1.0)] * 12, 'maxiter': 35, 'popsize': 18, 'seed': 777},
        {'bounds': [(-6.0, 4.0)] * 12, 'maxiter': 30, 'popsize': 20, 'seed': 999},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 60:
            break
        try:
            de_bounds = cfg['bounds']
            
            result_de = differential_evolution(
                _objective_log,
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
                candidate_log = result_de.x
                score = _objective_log(candidate_log)
                if score < best_score:
                    best_score = score
                    best_log_roots = candidate_log.copy()
            
            # Polish the DE result with tight tolerances in log-space
            if not np.any(np.isnan(result_de.x)):
                result_polish = minimize(
                    _objective_log,
                    result_de.x,
                    method='L-BFGS-B',
                    bounds=log_bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if not np.any(np.isnan(result_polish.x)):
                    candidate_log = np.clip(result_polish.x, -13.8, 4.6)
                    score = _objective_log(candidate_log)
                    if score < best_score:
                        best_score = score
                        best_log_roots = candidate_log.copy()
                        
        except Exception:
            pass
    
    # Final high-precision polish on best roots in log-space
    if time.time() < deadline - 10:
        try:
            for ftol, gtol, maxit in [(1e-14, 1e-12, 300), (1e-15, 1e-13, 300), (1e-16, 1e-14, 200)]:
                if time.time() > deadline - 5:
                    break
                result_final = minimize(
                    _objective_log,
                    best_log_roots,
                    method='L-BFGS-B',
                    bounds=log_bounds,
                    options={'maxiter': maxit, 'ftol': ftol, 'gtol': gtol}
                )
                if not np.any(np.isnan(result_final.x)):
                    candidate_log = np.clip(result_final.x, -13.8, 4.6)
                    score = _objective_log(candidate_log)
                    if score < best_score:
                        best_score = score
                        best_log_roots = candidate_log.copy()
        except Exception:
            pass
    
    best_roots = np.exp(best_log_roots)
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
