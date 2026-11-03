"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _compute_norm_log_and_grad(roots, n_quad=400):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)^2
    (double roots) with weight e^{-x}, and its gradient w.r.t. roots.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound.
    """
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - r_i)^2, so log|P(x)| = 2 * sum log|x - r_i|
    # log P(x)^2 = 4 * sum log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 4.0 * np.log(diff)
    
    # Integrand: e^{-x} * P(x)^2
    log_integrand = -x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf, np.zeros(m)
    
    exp_shifted = np.exp(log_integrand - max_log)
    weighted = w_quad * exp_shifted
    sum_val = np.sum(weighted)
    
    if sum_val <= 0:
        return -np.inf, np.zeros(m)
        
    log_norm_sq = max_log + np.log(sum_val)
    
    # Gradient: d log_norm_sq / d r_j = 4 * sum_i weighted_i * (1/(x_i - r_j)) / sum_val
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        diff[np.abs(diff) < 1e-300] = 1e-300
        grad[j] = 4.0 * np.sum(weighted / diff)
    
    return log_norm_sq, grad

def _get_laguerre_zeros(m=12):
    """Get zeros of Laguerre polynomial L_m as initialization."""
    try:
        x, _ = roots_laguerre(m)
        return np.sort(x)
    except Exception:
        return np.linspace(1.0, 40.0, m)

def _get_geometric_roots(m=12, ratio=1.25, start=3.0):
    """Generate geometrically spaced roots."""
    return start * (ratio ** np.arange(m))

def _get_random_roots(m=12, seed=0):
    """Generate random roots in a reasonable range."""
    rng = np.random.RandomState(seed)
    return np.sort(rng.uniform(1.0, 60.0, m))

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    m = 12
    
    # Generate diverse initializations
    inits = []
    
    # 1. Laguerre polynomial zeros (physically motivated)
    lag_roots = _get_laguerre_zeros(m)
    inits.append(lag_roots)
    
    # 2. Geometric spacing with different ratios
    for ratio in [1.15, 1.2, 1.25, 1.3, 1.4]:
        inits.append(_get_geometric_roots(m, ratio, 3.0))
    
    # 3. Linear spacing (seed)
    inits.append(np.array([4.0 * i for i in range(1, 13)]))
    
    # 4. Random initializations
    for seed in [42, 123, 456, 789]:
        inits.append(_get_random_roots(m, seed))
    
    # 5. Clustered roots (some close together)
    clustered = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 33.0, 42.0, 52.0, 63.0, 75.0])
    inits.append(clustered)
    
    # Evaluate all initializations
    best_roots = None
    best_log_norm = -np.inf
    
    for init in inits:
        if len(init) != m or np.any(init <= 0):
            continue
        log_norm, _ = _compute_norm_log_and_grad(np.sort(init))
        if np.isfinite(log_norm) and log_norm > best_log_norm:
            best_log_norm = log_norm
            best_roots = np.sort(init).copy()
    
    if best_roots is None:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        best_log_norm = _compute_norm_log_and_grad(best_roots)[0]
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return 1e10
        roots = np.sort(params)
        if np.any(roots <= 1e-6) or np.any(roots > 200):
            return 1e10
        log_norm, _ = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    def objective_grad(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return 1e10, np.zeros(m)
        roots = np.sort(params)
        if np.any(roots <= 1e-6) or np.any(roots > 200):
            return 1e10, np.zeros(m)
        log_norm, grad = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(m)
        return -log_norm, -grad
    
    # Multi-start local optimization
    # Try to optimize from several of the best initializations
    # Sort inits by their log_norm to prioritize good ones
    init_scores = []
    for init in inits:
        if len(init) != m or np.any(init <= 0):
            continue
        log_norm, _ = _compute_norm_log_and_grad(np.sort(init))
        if np.isfinite(log_norm):
            init_scores.append((log_norm, np.sort(init).copy()))
    
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Optimize from top 5 initializations
    for idx, (init_log_norm, init_roots) in enumerate(init_scores[:5]):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        try:
            result = minimize(
                objective, 
                init_roots, 
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=[(0.5, 150.0)] * m,
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 1e9:
                result_roots = np.sort(result.x)
                result_log_norm = -result.fun
                if np.isfinite(result_log_norm) and result_log_norm > best_log_norm:
                    best_log_norm = result_log_norm
                    best_roots = result_roots.copy()
                    
        except Exception:
            pass
    
    # Final polish on the best result
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result_final = minimize(
                objective, 
                best_roots, 
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=[(0.5, 150.0)] * m,
                options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            
            if result_final.success or result_final.fun < 1e9:
                result_roots = np.sort(result_final.x)
                result_log_norm = -result_final.fun
                if np.isfinite(result_log_norm) and result_log_norm > best_log_norm:
                    best_log_norm = result_log_norm
                    best_roots = result_roots.copy()
                    
        except Exception:
            pass
    
    # Final verification
    best_roots = np.sort(best_roots)
    if len(best_roots) != m or np.any(best_roots <= 0) or not np.all(np.isfinite(best_roots)):
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
