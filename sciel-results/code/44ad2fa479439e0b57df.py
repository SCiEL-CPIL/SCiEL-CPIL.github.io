"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_norm_log_and_grad(roots, n_quad=500):
    """
    Compute the log of the L2 norm squared of the polynomial P(x) = prod (x - r_i)^2
    (double roots) with weight e^{-x}, and its gradient w.r.t. roots.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound.
    
    P(x) = prod_i (x - r_i)^2
    |P(x)|^2 = prod_i (x - r_i)^4
    log |P(x)|^2 = 4 * sum_i log|x - r_i|
    """
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log |P(x)|^2 = 4 * sum log|x - r_i|
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

def _dyson_gas_init(m=12):
    """
    Initialize root positions using Dyson gas equilibrium.
    
    For the Laguerre weight e^{-x}, the equilibrium condition for roots of
    a polynomial with double roots is derived from the variational principle.
    The effective repulsion is stronger due to the double-root structure.
    
    We use the condition: for each root r_i,
    sum_{j != i} 4/(r_i - r_j) = r_i  (approximately, from the log-derivative)
    
    Actually, the correct equilibrium comes from d/dr_i log||P||^2 = 0 at optimum,
    which gives: 4 * sum_j w(x_j)/(x_j - r_i) * |P(x_j)|^2 e^{-x_j} / ||P||^2 = 0
    This is implicit, so we use a simpler heuristic initialization.
    """
    # Start from positions scaled to match Laguerre node spacing
    # Laguerre nodes for degree ~24 (since we have 12 double roots, total degree 24)
    try:
        x_nodes, _ = np.polynomial.laguerre.laggauss(24)
        # Take every other node as initial guess for 12 roots
        roots = x_nodes[::2].copy()
    except:
        roots = np.linspace(2.0, 45.0, m)
    
    # Refine with iterative relaxation
    for iteration in range(1000):
        forces = np.zeros(m)
        for i in range(m):
            forces[i] = roots[i]  # Confining force
            for j in range(m):
                if i != j:
                    forces[i] -= 4.0 / (roots[i] - roots[j])
        
        # Adaptive step size
        max_force = np.max(np.abs(forces))
        if max_force < 1e-12:
            break
        
        step = min(0.05, 0.5 / max_force)
        for i in range(m):
            roots[i] += step * forces[i]
            roots[i] = max(roots[i], 0.1)
    
    return np.sort(roots)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Use Dyson gas initialization for better starting point
    try:
        dyson_roots = _dyson_gas_init(12)
    except Exception:
        dyson_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Also keep the seed roots as a fallback
    seed_roots = np.array([4.0 * i for i in range(1, 13)])
    
    # Also try Laguerre node-based initialization
    try:
        x_nodes, _ = np.polynomial.laguerre.laggauss(24)
        lag_roots = x_nodes[::2].copy()
    except:
        lag_roots = seed_roots.copy()
    
    # Evaluate all starting points
    candidates = [dyson_roots, seed_roots, lag_roots]
    best_roots = candidates[0].copy()
    best_log_norm = -np.inf
    
    for cand in candidates:
        cand = np.sort(cand)
        if np.any(cand <= 0) or len(cand) != 12:
            continue
        ln, _ = _compute_norm_log_and_grad(cand)
        if ln > best_log_norm:
            best_log_norm = ln
            best_roots = cand.copy()
    
    # We want to MAXIMIZE the norm, so MINIMIZE the negative log norm
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            return 1e10  # Penalty for timeout
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10
        log_norm, _ = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    def objective_grad(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            return 1e10, np.zeros(len(params))
        roots = np.sort(params)
        if any(r <= 1e-6 for r in roots):
            return 1e10, np.zeros(len(params))
        log_norm, grad = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(len(params))
        return -log_norm, -grad  # Minimize negative log norm
    
    # Optimization with multiple strategies
    try:
        # Strategy 1: L-BFGS-B from best initial point
        result = minimize(objective, best_roots, jac=objective_grad,
                         method='L-BFGS-B',
                         bounds=[(0.1, None)] * 12,
                         options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-12})
        
        if result.fun < -best_log_norm:
            best_roots = np.sort(result.x)
            best_log_norm = -result.fun
            
        # Strategy 2: Try from each initial point if time permits
        for cand in candidates[1:]:
            if time.time() - start_time > deadline - 120:
                break
            cand = np.sort(cand)
            result2 = minimize(objective, cand, jac=objective_grad,
                             method='L-BFGS-B',
                             bounds=[(0.1, None)] * 12,
                             options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if result2.fun < -best_log_norm:
                best_roots = np.sort(result2.x)
                best_log_norm = -result2.fun
        
        # Strategy 3: Polish with tighter tolerances
        if time.time() - start_time < deadline - 60:
            result3 = minimize(objective, best_roots, jac=objective_grad,
                             method='L-BFGS-B',
                             bounds=[(0.1, None)] * 12,
                             options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-14})
            
            if result3.fun < -best_log_norm:
                best_roots = np.sort(result3.x)
                best_log_norm = -result3.fun
                
        # Strategy 4: Nelder-Mead for local refinement if time permits
        if time.time() - start_time < deadline - 30:
            result4 = minimize(objective, best_roots, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if result4.fun < -best_log_norm:
                best_roots = np.sort(result4.x)
                best_log_norm = -result4.fun
                
    except Exception:
        pass
    
    # Final verification: Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = seed_roots.copy()
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
