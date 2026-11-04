"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Gauss-Laguerre quadrature nodes and weights
_N_QUAD = 600
_x_quad, _w_quad = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_log_norm_and_grad(roots):
    """
    Compute the log of the L2 norm squared of P(x) = prod (x - r_i)^2
    with weight e^{-x}, and its gradient with respect to roots.
    We want to MAXIMIZE this norm.
    """
    roots = np.asarray(roots, dtype=np.float64)
    n_roots = len(roots)
    
    # Compute log P(x)^2 = 4 * sum log|x - r_i|
    log_p_sq = np.zeros(_N_QUAD)
    for r in roots:
        diff = np.abs(_x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 4.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
    log_integrand = -_x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf, np.zeros(n_roots)
    
    # Weighted contributions
    weights_exp = _w_quad * np.exp(log_integrand - max_log)
    sum_val = np.sum(weights_exp)
    
    if sum_val <= 0:
        return -np.inf, np.zeros(n_roots)
    
    log_norm_sq = max_log + np.log(sum_val)
    
    # Gradient: d/dr_j [log ||P||^2] = 4 * sum_i weights_exp_i / (x_i - r_j) / sum_val
    grad = np.zeros(n_roots)
    for j, r in enumerate(roots):
        diff = _x_quad - r
        # Avoid division by zero
        diff[diff < 1e-300] = 1e-300
        diff[diff > -1e-300] = -1e-300
        grad[j] = 4.0 * np.sum(weights_exp / diff) / sum_val
    
    return log_norm_sq, grad

def _compute_log_norm(roots):
    """Compute the log of the L2 norm squared."""
    val, _ = _compute_log_norm_and_grad(roots)
    return val

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_roots = None
    best_log_norm = -np.inf
    
    def objective_full(params):
        """Objective function for full 12-dimensional root optimization (minimize negative log norm)."""
        if time.time() > deadline - 10:
            return 1e10
        roots = np.sort(params)
        if np.any(roots <= 1e-6) or np.any(roots > 1000):
            return 1e10
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm

    def grad_full(params):
        """Gradient for full 12-dimensional root optimization."""
        if time.time() > deadline - 10:
            return np.zeros(12)
        roots = np.sort(params)
        if np.any(roots <= 1e-6) or np.any(roots > 1000):
            return np.zeros(12)
        _, grad = _compute_log_norm_and_grad(roots)
        # We're minimizing -log_norm, so gradient is -grad
        return -grad

    # Phase 1: Global search over 12 independent roots using Differential Evolution
    bounds = [(1e-3, 500.0)] * 12
    
    try:
        de_result = differential_evolution(
            objective_full, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-10,
            popsize=20,
            init='sobol',
            polish=False,
            updating='deferred'
        )
        
        if de_result.fun < 1e9:
            best_roots = np.sort(de_result.x)
            best_log_norm = _compute_log_norm(best_roots)
            
    except Exception:
        pass
    
    # Phase 2: Multi-start local refinement on 12 independent roots
    # Generate multiple starting points
    start_points = []
    
    # Start from DE result
    if best_roots is not None:
        start_points.append(best_roots.copy())
    
    # Add diverse geometric sequences
    for mu_try in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for sigma_try in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            i = np.arange(12)
            rp = np.exp(mu_try + sigma_try * i)
            if np.all(rp > 1e-6) and np.all(rp < 500):
                start_points.append(rp)
    
    # Add arithmetic sequences
    for start_val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for step in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rp = np.array([start_val + step * i for i in range(12)])
            if np.all(rp > 1e-6) and np.all(rp < 500):
                start_points.append(rp)
    
    # Add log-spaced sequences
    for start_log in [0.0, 0.5, 1.0, 1.5]:
        for end_log in [2.0, 2.5, 3.0, 3.5]:
            if end_log > start_log:
                rp = np.exp(np.linspace(start_log, end_log, 12))
                if np.all(rp > 1e-6) and np.all(rp < 500):
                    start_points.append(rp)
    
    # Keep track of best
    for sp in start_points:
        if time.time() > deadline - 60:
            break
        sp = np.sort(sp)
        try:
            result = minimize(
                objective_full, 
                sp, 
                jac=grad_full,
                method='L-BFGS-B',
                bounds=[(1e-6, 500.0)] * 12,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 100}
            )
            
            if result.fun < 1e9 and result.fun < -best_log_norm:
                best_roots = np.sort(result.x)
                best_log_norm = -result.fun
                
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_roots is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                objective_full, 
                best_roots, 
                jac=grad_full,
                method='L-BFGS-B',
                bounds=[(1e-6, 500.0)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 200}
            )
            
            if result.fun < 1e9 and result.fun < -best_log_norm:
                best_roots = np.sort(result.x)
                best_log_norm = -result.fun
                
        except Exception:
            pass
    
    # Final check
    if best_roots is None:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
    
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
