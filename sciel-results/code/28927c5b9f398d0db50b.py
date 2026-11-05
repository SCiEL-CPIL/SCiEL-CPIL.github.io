"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute quadrature points at high resolution
_N_QUAD = 2000
_X_QUAD, _W_QUAD = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_norm_log_and_grad(roots, x_quad=None, w_quad=None):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)^2
    (double roots) with weight e^{-x}, and its gradient w.r.t. roots.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound.
    
    P(x) = prod_i (x - r_i)^2
    |P(x)|^2 = prod_i (x - r_i)^4
    log |P(x)|^2 = 4 * sum_i log|x - r_i|
    """
    m = len(roots)
    if x_quad is None:
        x_quad = _X_QUAD
    if w_quad is None:
        w_quad = _W_QUAD
    
    # Compute log |P(x)|^2 = 4 * sum_i log|x - r_i|
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 4.0 * np.log(diff)
    
    # Integrand: e^{-x} * |P(x)|^2
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
    
    # Gradient: d/d(r_j) log_norm_sq = 4 * sum_i weighted_i * (1/(x_i - r_j))
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        # Avoid division by zero
        diff[np.abs(diff) < 1e-300] = 1e-300
        grad[j] = 4.0 * np.sum(weighted / diff)
    
    return log_norm_sq, grad

def _dyson_gas_init(m=12, repulsion=4.0):
    """
    Initialize root positions using Dyson gas equilibrium with repulsion exponent
    (matching double roots -> repulsion ~ 4) and linear confining potential.
    
    For double roots, the effective repulsion is stronger. The equilibrium:
    sum_{j != i} repulsion/(x_i - x_j) = x_i
    """
    # Start from uniform spacing
    roots = np.linspace(2.0, 50.0, m)
    
    for iteration in range(1000):
        # Compute forces
        forces = np.zeros(m)
        for i in range(m):
            # Confining force (from linear potential)
            forces[i] = roots[i]
            # Repulsive forces from other particles
            for j in range(m):
                if i != j:
                    forces[i] -= repulsion / (roots[i] - roots[j])
        
        # Update positions with adaptive step
        max_force = np.max(np.abs(forces))
        step_size = min(0.1, 1.0 / (1.0 + max_force))
        for i in range(m):
            roots[i] += step_size * forces[i]
            roots[i] = max(roots[i], 0.5)  # Keep positive
        
        # Check convergence
        if max_force < 1e-12:
            break
    
    return np.sort(roots)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Use Dyson gas initialization with correct repulsion for double roots
    try:
        dyson_roots = _dyson_gas_init(12, repulsion=4.0)
    except Exception:
        dyson_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Also keep the seed roots as a fallback
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Generate diverse initializations
    initializations = []
    initializations.append(dyson_roots)
    initializations.append(seed_roots)
    
    # Add perturbed versions for multi-start
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = seed_roots * (1.0 + 0.1 * rng.randn(12))
        perturbed = np.sort(np.maximum(perturbed, 0.5))
        initializations.append(perturbed)
    
    # Evaluate all initializations and pick the best
    best_roots = initializations[0].copy()
    best_log_norm, _ = _compute_norm_log_and_grad(best_roots)
    
    for init in initializations[1:]:
        log_norm, _ = _compute_norm_log_and_grad(init)
        if log_norm > best_log_norm:
            best_roots = init.copy()
            best_log_norm = log_norm
    
    # We want to MAXIMIZE the norm, so MINIMIZE the negative log norm
    # Use log-space parameterization to ensure positivity
    def _log_space_objective(log_params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            return 1e10
        roots = np.exp(log_params)
        roots = np.sort(roots)
        if any(r <= 1e-6 or r > 1e6 for r in roots):
            return 1e10
        log_norm, _ = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    def _log_space_objective_grad(log_params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            return 1e10, np.zeros(len(log_params))
        roots = np.exp(log_params)
        roots = np.sort(roots)
        if any(r <= 1e-6 or r > 1e6 for r in roots):
            return 1e10, np.zeros(len(log_params))
        log_norm, grad = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(len(log_params))
        # Chain rule: d/d(log_r_j) = r_j * d/d(r_j)
        grad_log = grad * roots
        return -log_norm, -grad_log
    
    # Optimization with multiple strategies
    try:
        # Strategy 1: L-BFGS-B in log-space from best initial point
        log_init = np.log(best_roots)
        result = minimize(_log_space_objective, log_init, jac=_log_space_objective_grad,
                         method='L-BFGS-B',
                         bounds=[(np.log(0.5), np.log(200.0))] * 12,
                         options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if result.success:
            candidate_roots = np.sort(np.exp(result.x))
            cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
            if cand_log_norm > best_log_norm:
                best_roots = candidate_roots
                best_log_norm = cand_log_norm
        
        # Strategy 2: Try from other initializations if time permits
        for init in initializations[1:4]:
            if time.time() - start_time < deadline - 120:
                log_init2 = np.log(init)
                result2 = minimize(_log_space_objective, log_init2, jac=_log_space_objective_grad,
                                 method='L-BFGS-B',
                                 bounds=[(np.log(0.5), np.log(200.0))] * 12,
                                 options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if result2.success:
                    candidate_roots = np.sort(np.exp(result2.x))
                    cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                    if cand_log_norm > best_log_norm:
                        best_roots = candidate_roots
                        best_log_norm = cand_log_norm
            else:
                break
        
        # Strategy 3: Polish with tighter tolerances
        if time.time() - start_time < deadline - 60:
            log_best = np.log(best_roots)
            result3 = minimize(_log_space_objective, log_best, jac=_log_space_objective_grad,
                             method='L-BFGS-B',
                             bounds=[(np.log(0.5), np.log(200.0))] * 12,
                             options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-14})
            
            if result3.success:
                candidate_roots = np.sort(np.exp(result3.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
                
        # Strategy 4: Nelder-Mead for local refinement if time permits
        if time.time() - start_time < deadline - 30:
            result4 = minimize(_log_space_objective, np.log(best_roots), method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if result4.success:
                candidate_roots = np.sort(np.exp(result4.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
                    
    except Exception:
        pass
    
    # Final verification: Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = seed_roots.copy()
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
