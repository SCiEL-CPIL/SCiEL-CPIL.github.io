"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Gauss-Laguerre quadrature points
N_QUAD = 200
x_quad, w_quad = np.polynomial.laguerre.laggauss(N_QUAD)
exp_neg_x = np.exp(-x_quad)
w_exp = w_quad * exp_neg_x

def _compute_c4_and_grad(roots):
    """
    Compute C4 and its gradient with respect to roots.
    roots: array of shape (12,)
    Returns: (c4, grad)
    """
    roots = np.asarray(roots, dtype=np.float64)
    m = len(roots)
    
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros(m)
    
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    
    P = np.ones(N_QUAD)
    for i in range(m):
        P *= (x_quad - roots[i])**2
        
    P2 = P * P
    
    int_P = np.sum(w_exp * P)
    int_P2 = np.sum(w_exp * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf, np.zeros(m)
        
    c4 = int_P2 / (int_P * int_P)
    
    grad = np.zeros(m)
    eps = 1e-10
    mask = np.abs(diff) < eps
    inv_diff = np.zeros_like(diff)
    np.divide(1.0, diff, out=inv_diff, where=~mask)
    
    dI0 = 2.0 * ((w_exp * P) @ inv_diff)
    dI1 = 4.0 * ((w_exp * P2) @ inv_diff)
    
    denom = int_P**3
    grad = (dI1 * int_P - 2.0 * int_P2 * dI0) / denom
    
    return c4, grad

def _objective_and_grad(y):
    """
    Objective and gradient for L-BFGS-B.
    y: log(roots)
    """
    roots = np.exp(y)
    c4, grad_roots = _compute_c4_and_grad(roots)
    
    if np.isinf(c4):
        return c4, np.zeros_like(y)
        
    grad_y = grad_roots / roots
    return c4, grad_y

def _concave_roots(log_a, log_b, curvature):
    """
    Generate 12 roots from a concave ansatz.
    roots[i] = exp(log_a + (log_b - log_a) * t_i) where t_i are concave-spaced
    """
    n = 12
    t_lin = np.linspace(0, 1, n)
    t_concave = t_lin ** curvature
    t_concave = np.clip(t_concave, 1e-6, 1 - 1e-6)
    log_roots = log_a + (log_b - log_a) * t_concave
    return np.exp(log_roots)

def _objective_3param(params):
    """
    Objective for 3-parameter optimization: (log_a, log_b, curvature)
    """
    log_a, log_b, curvature = params
    if log_a >= log_b:
        return 1e10
    if curvature <= 0.1 or curvature >= 10:
        return 1e10
    
    roots = _concave_roots(log_a, log_b, curvature)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e10
    
    c4, _ = _compute_c4_and_grad(roots)
    return c4 if np.isfinite(c4) else 1e10

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580
    
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score, _ = _compute_c4_and_grad(best_roots)
    
    # Phase 1: Global search with Differential Evolution on 3-parameter ansatz
    if time.time() < deadline - 300:
        try:
            bounds_de = [(-5.0, 2.0), (0.0, 5.0), (0.2, 5.0)]
            
            de_result = differential_evolution(
                _objective_3param,
                bounds_de,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=15,
                polish=False
            )
            
            if de_result.fun < best_score:
                log_a, log_b, curvature = de_result.x
                roots_de = _concave_roots(log_a, log_b, curvature)
                roots_de = np.sort(roots_de)
                score_de, _ = _compute_c4_and_grad(roots_de)
                if score_de < best_score:
                    best_score = score_de
                    best_roots = roots_de.copy()
                    
        except Exception as e:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B on 12-dim log-space
    if time.time() < deadline - 60:
        try:
            y_init = np.log(best_roots)
            bounds = [(np.log(1e-6), np.log(1000))] * 12
            
            result = minimize(
                _objective_and_grad,
                y_init,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success and result.fun < best_score:
                y_opt = result.x
                x_opt = np.exp(y_opt)
                x_opt = np.sort(x_opt)
                score, _ = _compute_c4_and_grad(x_opt)
                if score < best_score:
                    best_score = score
                    best_roots = x_opt.copy()
                    
        except Exception as e:
            pass
    
    # Phase 3: Multi-start local refinement from diverse initial points
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        
        initial_points = [np.log(best_roots)]
        
        for _ in range(5):
            noise = rng.normal(0, 0.3, 12)
            initial_points.append(np.log(best_roots) + noise)
        
        initial_points.append(np.log(x0))
        for scale in [1.5, 2.0]:
            initial_points.append(np.log(x0 * scale))
        
        bounds = [(np.log(1e-6), np.log(1000))] * 12
        
        for i, y_init in enumerate(initial_points):
            if time.time() > deadline - 10:
                break
                
            try:
                result = minimize(
                    _objective_and_grad,
                    y_init,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if result.success and result.fun < best_score:
                    y_opt = result.x
                    x_opt = np.exp(y_opt)
                    x_opt = np.sort(x_opt)
                    score, _ = _compute_c4_and_grad(x_opt)
                    if score < best_score:
                        best_score = score
                        best_roots = x_opt.copy()
                        
            except Exception as e:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            y_best = np.log(best_roots)
            result_final = minimize(
                _objective_and_grad,
                y_best,
                jac=True,
                method='L-BFGS-B',
                bounds=[(np.log(1e-6), np.log(1000))] * 12,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if result_final.fun < best_score:
                y_opt = result_final.x
                x_opt = np.exp(y_opt)
                x_opt = np.sort(x_opt)
                score, _ = _compute_c4_and_grad(x_opt)
                if score < best_score:
                    best_score = score
                    best_roots = x_opt.copy()
        except Exception:
            pass

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
