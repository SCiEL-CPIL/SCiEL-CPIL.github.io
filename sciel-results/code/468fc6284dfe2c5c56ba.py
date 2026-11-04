"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

# Precompute quadrature nodes and weights
_N_QUAD = 200
_X_QUAD, _W_QUAD = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # P(x) = prod (x - x_i)^2
    # C4 = [∫ P(x)^2 e^{-x} dx] / [∫ P(x) e^{-x} dx]^2
    
    # Use log-space for numerical stability
    log_P = np.zeros(_N_QUAD)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(_X_QUAD - r) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(_W_QUAD * P)
    int_P2 = np.sum(_W_QUAD * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _compute_c4_bound_and_gradient(log_params):
    """Compute C4 bound and gradient with respect to log-parameters."""
    roots = np.exp(log_params)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e100, np.zeros(12)
    
    # Compute P(x) and dP/dx_i
    P = np.ones(_N_QUAD)
    dP_droots = np.zeros((12, _N_QUAD))
    
    for i, r in enumerate(roots):
        diff = _X_QUAD - r
        P *= diff * diff
        # dP/dr_i = P * 2 * (x - r_i) / (x - r_i)^2 = P * 2 / (x - r_i)
        # More carefully: P = prod_j (x - r_j)^2
        # dP/dr_i = P * sum_j d/d r_i log((x - r_j)^2) = P * 2 * d/d r_i log(x - r_i)
        # = P * 2 * (-1/(x - r_i)) = -2P/(x - r_i)
        dP_droots[i] = -2.0 * P / (diff + 1e-300)
    
    P2 = P * P
    int_P = np.sum(_W_QUAD * P)
    int_P2 = np.sum(_W_QUAD * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return 1e100, np.zeros(12)
    
    c4 = int_P2 / (int_P * int_P)
    
    # Gradient: dc4/dr_i = d/dr_i [int_P2 / int_P^2]
    # = (dint_P2/dr_i * int_P^2 - int_P2 * 2 * int_P * dint_P/dr_i) / int_P^4
    # = (dint_P2/dr_i * int_P - 2 * int_P2 * dint_P/dr_i) / int_P^3
    
    dint_P = np.array([np.sum(_W_QUAD * dP_droots[i]) for i in range(12)])
    dint_P2 = np.array([2.0 * np.sum(_W_QUAD * P * dP_droots[i]) for i in range(12)])
    
    grad_c4_dr = (dint_P2 * int_P - 2.0 * int_P2 * dint_P) / (int_P ** 3)
    
    # Chain rule: dc4/d(log r_i) = dc4/dr_i * r_i
    grad_c4_dlogp = grad_c4_dr * roots
    
    return c4, grad_c4_dlogp

def _simulated_annealing(log_params, n_iter=5000, T0=1.0, T1=0.01, cooling=0.9998):
    """Simulated annealing in log-space."""
    best_params = log_params.copy()
    best_score = _compute_c4_bound(np.exp(log_params))
    current_params = log_params.copy()
    current_score = best_score
    T = T0
    
    for i in range(n_iter):
        # Random perturbation
        delta = np.random.randn(12) * 0.1
        new_params = current_params + delta
        new_roots = np.exp(new_params)
        
        if np.any(new_roots <= 0) or np.any(np.isnan(new_roots)):
            T *= cooling
            continue
        
        new_score = _compute_c4_bound(new_roots)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_params = new_params.copy()
        
        T *= cooling
    
    return best_params, best_score

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial guess: Standard Laguerre roots
    x0, _ = roots_laguerre(12)
    best_roots = x0.copy()
    best_score = _compute_c4_bound(best_roots)
    best_log_params = np.log(x0.copy())
    
    # Phase 1: Simulated annealing for global exploration
    n_restarts = 5
    sa_iterations = 3000
    
    for restart in range(n_restarts):
        if time.time() > deadline - 120:
            break
        
        # Initialize from best so far with some noise
        if restart == 0:
            init_log_params = best_log_params.copy()
        else:
            init_log_params = best_log_params + np.random.randn(12) * 0.5
        
        try:
            sa_params, sa_score = _simulated_annealing(
                init_log_params, 
                n_iter=sa_iterations,
                T0=1.0,
                T1=0.01,
                cooling=0.9999
            )
            
            if sa_score < best_score:
                best_score = sa_score
                best_roots = np.exp(sa_params)
                best_log_params = sa_params.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement with analytical gradients
    if time.time() < deadline - 30:
        try:
            def _objective_with_grad(log_params):
                return _compute_c4_bound_and_gradient(log_params)
            
            # Multiple L-BFGS-B restarts from different points
            n_lfbs_restarts = 3
            for lfbs_restart in range(n_lfbs_restarts):
                if time.time() > deadline - 30:
                    break
                
                if lfbs_restart == 0:
                    init = best_log_params.copy()
                else:
                    init = best_log_params + np.random.randn(12) * 0.1
                
                result = minimize(
                    _objective_with_grad,
                    init,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                if result.fun < best_score:
                    candidate_roots = np.exp(result.x)
                    if np.all(candidate_roots > 0) and not np.any(np.isnan(candidate_roots)):
                        candidate_roots = np.sort(candidate_roots)
                        score = _compute_c4_bound(candidate_roots)
                        if score < best_score:
                            best_score = score
                            best_roots = candidate_roots.copy()
                            best_log_params = result.x.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
