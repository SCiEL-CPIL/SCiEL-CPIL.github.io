"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def _compute_c4_bound_fast(roots):
    """
    Compute the C4 upper bound for given double-root positions.
    Uses Gauss-Laguerre quadrature for numerical integration.
    Metric: int(P^2 * w) / (int(P * w))^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature (includes e^{-x} weight)
    # 150 points is a good balance of accuracy and speed
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2
    # Compute log_P for numerical stability
    # Vectorized: diff has shape (n_quad, 12)
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

def _roots_from_params(params):
    """
    Generate 12 positive double-root locations from 4 parameters.
    Uses a smooth parametric curve: r_k = A * exp(B * k) + C * sin(D * k + phase)
    where k = 0, 1, ..., 11. Ensures positivity by construction.
    """
    A, B, C, D = params
    k = np.arange(12, dtype=np.float64)
    # Base exponential growth
    base = A * np.exp(B * k)
    # Add sinusoidal modulation for non-geometric density
    mod = C * np.sin(D * k + 0.5)
    roots = base + mod
    # Ensure positivity
    roots = np.maximum(roots, 1e-6)
    # Ensure monotonicity (sort)
    roots = np.sort(roots)
    return roots

def _objective_params(params):
    """Objective function over 4 parameters."""
    roots = _roots_from_params(params)
    return _compute_c4_bound_fast(roots)

def _fit_params_to_roots(target_roots):
    """
    Given 12 target roots, fit 4 parameters (A, B, C, D) that best reproduce them.
    Uses least squares on the log-space representation.
    """
    target_roots = np.asarray(target_roots, dtype=np.float64)
    k = np.arange(12, dtype=np.float64)
    
    def residual(params):
        A, B, C, D = params
        predicted = A * np.exp(B * k) + C * np.sin(D * k + 0.5)
        return predicted - target_roots
    
    # Initial guess from log-linear fit
    log_roots = np.log(np.maximum(target_roots, 1e-10))
    # Fit log(r_k) ≈ log(A) + B*k for initial A, B
    A_init = np.exp(np.mean(log_roots))
    B_init = np.polyfit(k, log_roots, 1)[0] if len(k) > 1 else 0.1
    C_init = 0.1 * A_init
    D_init = 0.5
    
    p0 = [A_init, B_init, C_init, D_init]
    
    try:
        result = minimize(
            lambda p: np.sum(residual(p)**2),
            p0,
            method='L-BFGS-B',
            bounds=[(1e-6, 100), (-2.0, 2.0), (-50, 50), (-5.0, 5.0)],
            options={'maxiter': 200}
        )
        return result.x
    except Exception:
        return p0

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Strategy:
    1. Use standard Laguerre roots as a strong initial guess.
    2. Optimize in 4D parameter space (smooth parametric family).
    3. Also optimize in full 12D space for refinement.
    4. Return the best set of roots found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Initial guess: Zeros of Laguerre L_12(x)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Optimize in 4D parameter space
    # Fit parameters to Laguerre roots as starting point
    p0 = _fit_params_to_roots(x0)
    
    # Multiple random restarts in 4D
    for restart in range(20):
        if time.time() > deadline - 180:
            break
        
        if restart == 0:
            p_init = p0.copy()
        else:
            # Perturb the best parameters or random init
            if restart % 2 == 0:
                p_init = p0 + 0.3 * rng.standard_normal(4)
            else:
                p_init = [
                    rng.uniform(0.5, 5.0),   # A
                    rng.uniform(-0.5, 1.0),  # B
                    rng.uniform(-2.0, 2.0),  # C
                    rng.uniform(-2.0, 2.0)   # D
                ]
        
        try:
            result = minimize(
                _objective_params,
                p_init,
                method='L-BFGS-B',
                bounds=[(1e-4, 50), (-1.5, 1.5), (-30, 30), (-4.0, 4.0)],
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < best_score:
                candidate_roots = _roots_from_params(result.x)
                if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                    score = _compute_c4_bound_fast(candidate_roots)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate_roots.copy()
                        p0 = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Full 12D refinement from best 4D result
    if time.time() < deadline - 60:
        bounds = [(1e-6, None)] * 12
        try:
            result_final = minimize(
                lambda r: _compute_c4_bound_fast(r),
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 15:
        bounds = [(1e-8, None)] * 12
        try:
            result_polish = minimize(
                lambda r: _compute_c4_bound_fast(r),
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_polish.x > 0) and not np.any(np.isnan(result_polish.x)):
                candidate = np.sort(result_polish.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Ensure the output is a list of 12 positive floats
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        # Fallback to Laguerre roots if something went wrong
        result = [float(xi) for xi in x0]
        
    return result
# EVOLVE-BLOCK-END
