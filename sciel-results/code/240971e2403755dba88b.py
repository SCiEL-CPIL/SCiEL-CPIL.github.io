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
    
    # Gradient: d/d(r_j) log_norm_sq
    # d log|P|^2 / dr_j = 4 * d/d(r_j) sum_i log|x_i - r_j| (for the j-th term)
    # = 4 * (-1/(x_i - r_j)) for each quadrature point i
    # So d log_norm_sq / dr_j = 4 * sum_i weighted_i * (-1/(x_i - r_j)) / sum_val
    # Wait, let me re-derive:
    # log_norm_sq = max_log + log(sum_i w_i exp(log_integrand_i - max_log))
    # d log_norm_sq / dr_j = (1/sum_val) * sum_i w_i exp(log_integrand_i - max_log) * d(log_integrand_i)/dr_j
    # d(log_integrand_i)/dr_j = 4 * d(log|x_i - r_j|)/dr_j = 4 * (-1/(x_i - r_j))  [for i != j]
    # Actually: d(log|x - r|)/dr = -1/(x - r) when x > r, or 1/(x-r) when x < r
    # More carefully: d/d(r) log|x - r| = -sign(x-r)/(x-r) = -1/|x-r| * sign(x-r)... 
    # No: d/d(r) log|x-r| = d/d(r) log((x-r)^2)/2 = (1/2) * 2*(-1)/(x-r) = -1/(x-r)
    # So d(log|x-r|)/dr = -1/(x-r) always (since log|x-r| = log|x-r|)
    # Actually for x != r: d/d(r) ln|x-r| = -1/(x-r)
    
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        # Avoid division by zero
        diff[np.abs(diff) < 1e-300] = 1e-300
        grad[j] = 4.0 * np.sum(weighted / diff) * (-1.0)
    
    return log_norm_sq, grad

def _dyson_gas_init(m=12, beta=4.0):
    """
    Initialize root positions using Dyson gas equilibrium.
    For double roots with weight e^{-x}, the effective repulsion is stronger.
    
    The equilibrium condition for the energy E = -4 sum_{i<j} log|r_i - r_j| + sum_i r_i:
    dE/dr_i = 4 sum_{j != i} 1/(r_i - r_j) + r_i = 0
    => sum_{j != i} 1/(r_i - r_j) = -r_i/4
    """
    # Start from geometric progression-like spacing
    roots = np.array([3.0 * (1.25 ** i) for i in range(m)])
    roots = np.sort(roots)
    
    for iteration in range(1000):
        forces = np.zeros(m)
        for i in range(m):
            forces[i] = roots[i]
            for j in range(m):
                if i != j:
                    forces[i] -= 4.0 / (roots[i] - roots[j])
        
        step_size = 0.005
        for i in range(m):
            roots[i] += step_size * forces[i]
            roots[i] = max(roots[i], 0.3)
        
        if np.max(np.abs(forces)) < 1e-12:
            break
    
    return np.sort(roots)

def _lognormal_roots(params):
    """Generate roots as quantiles of a log-normal distribution."""
    mu, sigma = params
    # Use quantiles of the log-normal distribution
    # For m=12, use equal-probability quantiles
    quantiles = (np.arange(m) + 0.5) / m  # midpoints of equal probability bins
    # Log-normal quantile: exp(mu + sigma * z) where z = Phi^{-1}(q)
    from scipy.stats import norm
    z = norm.ppf(quantiles)
    roots = np.exp(mu + sigma * z)
    return np.sort(roots)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 595.0
    
    m = 12
    
    # Generate multiple initial candidates
    candidates = []
    
    # Candidate 1: Dyson gas initialization
    try:
        dyson_roots = _dyson_gas_init(m, beta=4.0)
        candidates.append(dyson_roots)
    except Exception:
        pass
    
    # Candidate 2: Geometric progression
    for ratio in [1.2, 1.25, 1.3, 1.35, 1.4, 1.5]:
        for start in [2.0, 3.0, 4.0, 5.0]:
            roots = np.array([start * (ratio ** i) for i in range(m)])
            candidates.append(np.sort(roots))
    
    # Candidate 3: Log-normal quantile parametrization with various parameters
    for mu in [np.log(4.0), np.log(5.0), np.log(6.0), np.log(7.0), np.log(8.0)]:
        for sigma in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
            try:
                roots = _lognormal_roots([mu, sigma])
                if all(r > 0.1 for r in roots):
                    candidates.append(roots)
            except Exception:
                pass
    
    # Candidate 4: Linear spacing
    for start in [2.0, 3.0, 4.0]:
        for end in [30.0, 40.0, 50.0]:
            roots = np.linspace(start, end, m)
            candidates.append(roots)
    
    # Evaluate all candidates and find the best
    best_roots = None
    best_log_norm = -np.inf
    
    for cand in candidates:
        if len(cand) != m:
            continue
        cand = np.sort(cand)
        if any(r <= 0.1 for r in cand):
            continue
        log_norm, _ = _compute_norm_log_and_grad(cand)
        if np.isfinite(log_norm) and log_norm > best_log_norm:
            best_log_norm = log_norm
            best_roots = cand.copy()
    
    if best_roots is None:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        best_log_norm, _ = _compute_norm_log_and_grad(best_roots)
    
    # Check time remaining
    if time.time() - start_time > deadline - 60:
        return best_roots.tolist()
    
    # Optimize each of the top candidates
    # First, sort candidates by their log_norm and take top few
    cand_scores = []
    for cand in candidates:
        if len(cand) != m or any(r <= 0.1 for r in cand):
            continue
        log_norm, _ = _compute_norm_log_and_grad(cand)
        if np.isfinite(log_norm):
            cand_scores.append((log_norm, cand))
    
    cand_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 distinct candidates for optimization
    top_candidates = []
    seen = set()
    for score, cand in cand_scores:
        key = tuple(np.round(cand / 0.5).astype(int))
        if key not in seen:
            seen.add(key)
            top_candidates.append(cand)
        if len(top_candidates) >= 5:
            break
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15:
            return 1e10
        roots = np.sort(params)
        if any(r <= 0.1 for r in roots):
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
        if any(r <= 0.1 for r in roots):
            return 1e10, np.zeros(len(params))
        log_norm, grad = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(len(params))
        return -log_norm, -grad
    
    # Optimize top candidates
    for i, init_roots in enumerate(top_candidates):
        if time.time() - start_time > deadline - 30:
            break
        
        try:
            result = minimize(objective, init_roots, jac=objective_grad,
                           method='L-BFGS-B',
                           bounds=[(0.1, None)] * m,
                           options={'maxiter': 500, 'maxfun': 15000, 'ftol': 1e-15, 'gtol': 1e-11})
            
            if result.fun < -best_log_norm:
                best_roots = np.sort(result.x)
                best_log_norm = -result.fun
        except Exception:
            pass
    
    # Polish with tighter tolerances on the best result
    if time.time() - start_time < deadline - 20:
        try:
            result2 = minimize(objective, best_roots, jac=objective_grad,
                           method='L-BFGS-B',
                           bounds=[(0.1, None)] * m,
                           options={'maxiter': 1000, 'maxfun': 30000, 'ftol': 1e-16, 'gtol': 1e-13})
            
            if result2.fun < -best_log_norm:
                best_roots = np.sort(result2.x)
                best_log_norm = -result2.fun
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() - start_time < deadline - 10:
        try:
            # Use even tighter tolerances
            def objective_tight(params):
                roots = np.sort(params)
                if any(r <= 0.1 for r in roots):
                    return 1e10
                log_norm, _ = _compute_norm_log_and_grad(roots, n_quad=700)
                if not np.isfinite(log_norm):
                    return 1e10
                return -log_norm
            
            def objective_grad_tight(params):
                roots = np.sort(params)
                if any(r <= 0.1 for r in roots):
                    return 1e10, np.zeros(len(params))
                log_norm, grad = _compute_norm_log_and_grad(roots, n_quad=700)
                if not np.isfinite(log_norm):
                    return 1e10, np.zeros(len(params))
                return -log_norm, -grad
            
            result3 = minimize(objective_tight, best_roots, jac=objective_grad_tight,
                           method='L-BFGS-B',
                           bounds=[(0.1, None)] * m,
                           options={'maxiter': 2000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-14})
            
            if result3.fun < -best_log_norm:
                best_roots = np.sort(result3.x)
                best_log_norm = -result3.fun
        except Exception:
            pass
    
    # Final verification
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != m:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
