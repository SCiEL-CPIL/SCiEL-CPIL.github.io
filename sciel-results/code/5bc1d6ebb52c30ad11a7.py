"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre, genlaguerre
from scipy.optimize import minimize, minimize_scalar
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
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2
    # Compute log_P for numerical stability
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

def _get_generalized_laguerre_roots(n, alpha):
    """
    Get the roots of the generalized Laguerre polynomial L_n^alpha(x).
    Returns None if it fails.
    """
    try:
        # scipy.special.genlaguerre returns a poly1d object
        poly = genlaguerre(n, alpha)
        coeffs = poly.coeffs
        rts = np.roots(coeffs)
        # Keep only real positive roots
        rts = rts[np.abs(rts.imag) < 1e-10]
        rts = np.real(rts)
        rts = rts[rts > 0]
        
        if len(rts) != n:
            return None
            
        rts = np.sort(rts)
        return rts
    except Exception:
        return None

def _beta_objective(beta):
    """
    Objective for optimizing beta: compute C4 bound for generalized Laguerre roots.
    This allows exploring a family of root configurations parameterized by the
    weight exponent in the orthogonality relation.
    """
    if beta < -0.9:
        return 1e10
    rts = _get_generalized_laguerre_roots(12, beta)
    if rts is None:
        return 1e10
    return _compute_c4_bound_fast(rts)

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Strategy:
    1. Use standard Laguerre roots as a strong initial guess.
    2. Explore the family of generalized Laguerre roots L_12^beta to find a better
       structural starting point.
    3. Generate diverse initial points (perturbations, geometric, random).
    4. Refine each with L-BFGS-B local optimization.
    5. Return the best set of roots found within the time limit.
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
    
    # Phase 1: Optimize beta for generalized Laguerre roots
    # This reduces the search to 1D while maintaining structure
    # The generalized Laguerre roots L_n^alpha are known to be optimal for certain
    # weighted moment problems, so this is a strong prior.
    try:
        # Coarse grid search first to find a good region
        beta_candidates = []
        for beta in np.linspace(-0.5, 5.0, 20):
            rts = _get_generalized_laguerre_roots(12, beta)
            if rts is not None:
                score = _compute_c4_bound_fast(rts)
                beta_candidates.append((beta, score, rts))
        
        if beta_candidates:
            # Sort by score and take top few
            beta_candidates.sort(key=lambda x: x[1])
            best_beta, best_beta_score, best_beta_roots = beta_candidates[0]
            
            if best_beta_score < best_score:
                best_score = best_beta_score
                best_roots = best_beta_roots.copy()
            
            # Refine beta around the best candidate
            beta_bounds = (max(best_beta - 0.5, -0.9), best_beta + 0.5)
            beta_result = minimize_scalar(
                _beta_objective,
                bounds=beta_bounds,
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 100}
            )
            
            if beta_result.success:
                refined_beta = beta_result.x
                refined_roots = _get_generalized_laguerre_roots(12, refined_beta)
                if refined_roots is not None:
                    refined_score = _compute_c4_bound_fast(refined_roots)
                    if refined_score < best_score:
                        best_score = refined_score
                        best_roots = refined_roots.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    # Phase 2: Generate diverse initial points for local optimization
    initial_points = [best_roots.copy()]
    
    # 1. Perturbed current best roots
    for scale in [0.01, 0.05, 0.1, 0.2]:
        perturbed = best_roots * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
        
    # 2. Standard Laguerre roots (in case they're better)
    initial_points.append(x0.copy())
    
    # 3. Scaled current best roots
    for scale in [0.8, 1.2, 1.5]:
        scaled = best_roots * scale
        initial_points.append(scaled.copy())
        
    # 4. Random uniform in [0.1, 30]
    for _ in range(5):
        rand_roots = rng.uniform(0.1, 30, size=12)
        rand_roots = np.sort(rand_roots)
        initial_points.append(rand_roots)
        
    # 5. Geometric spacing variations
    for start in [0.5, 1.0, 2.0]:
        for ratio in [1.3, 1.5, 1.8]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 100 and geom[0] > 0.1:
                initial_points.append(geom.copy())
                
    # 6. Log-uniform spacing
    for log_start in [-2.0, -1.0, 0.0]:
        for log_end in [1.0, 2.0, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 100:
                    initial_points.append(roots)

    # Phase 3: Refine each initial point with L-BFGS-B
    bounds = [(1e-6, None)] * 12
    
    for i, x_init in enumerate(initial_points):
        if time.time() > deadline - 60:
            break
        
        try:
            # Use L-BFGS-B with bounds to keep roots positive
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
    
    # Phase 4: Final polish on the best roots with tighter tolerance
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
    
    # Ensure the output is a list of 12 positive floats
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        # Fallback to Laguerre roots if something went wrong
        result = [float(xi) for xi in x0]
        
    return result
# EVOLVE-BLOCK-END
