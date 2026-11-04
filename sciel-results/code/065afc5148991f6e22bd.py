"""Refined Cohn-Goncalves Laguerre double-root construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute quadrature points once (higher resolution for accuracy)
_N_QUAD = 1000
_X_QUAD, _W_QUAD = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_norm_log_and_grad(roots, x_quad=None, w_quad=None):
    """
    Compute the log of the L2 norm squared of P(x) = prod (x - r_i)^2
    with weight e^{-x}, and its gradient w.r.t. roots.
    We want to MAXIMIZE this norm to MINIMIZE the C4 bound.
    
    P(x) = prod_i (x - r_i)^2
    |P(x)|^2 = prod_i (x - r_i)^4
    log |P(x)|^2 = 4 * sum_i log|x - r_i|
    """
    if x_quad is None:
        x_quad = _X_QUAD
    if w_quad is None:
        w_quad = _W_QUAD
    
    m = len(roots)
    
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
    """
    roots = np.linspace(2.0, 50.0, m)
    
    for iteration in range(2000):
        forces = np.zeros(m)
        for i in range(m):
            forces[i] = roots[i]
            for j in range(m):
                if i != j:
                    forces[i] -= repulsion / (roots[i] - roots[j])
        
        max_force = np.max(np.abs(forces))
        step_size = min(0.1, 1.0 / (1.0 + max_force))
        for i in range(m):
            roots[i] += step_size * forces[i]
            roots[i] = max(roots[i], 0.5)
        
        if max_force < 1e-14:
            break
    
    return np.sort(roots)

def _jacobian_eigen_init(m=12, alpha=1.0, beta=1.0):
    """
    Initialize using eigenvalues of a Jacobi matrix related to Laguerre polynomials.
    The Jacobi matrix for Laguerre polynomials with parameters alpha, beta has
    eigenvalues that are the zeros of the orthogonal polynomial.
    We modify the recurrence to get a different set of roots.
    """
    # For Laguerre polynomials L_m^{(alpha)}, the Jacobi matrix has:
    # a_i = 2*i + alpha + 1 (diagonal)
    # b_i = sqrt(i*(i + alpha)) (off-diagonal)
    # But we want to tune these to optimize the norm.
    
    # Start with standard Laguerre zeros and perturb
    # Standard Laguerre L_12 zeros
    from numpy.polynomial.laguerre import laggauss
    x_std, w_std = laggauss(m)
    
    # Create a modified Jacobi matrix
    J = np.zeros((m, m))
    for i in range(m):
        J[i, i] = 2.0 * (i + 1) + alpha + 1.0
    for i in range(m - 1):
        b = np.sqrt((i + 1) * (i + 1 + alpha)) * (1.0 + beta * 0.1)
        J[i, i+1] = b
        J[i+1, i] = b
    
    # Eigenvalues of symmetric matrix are real
    eigvals = np.linalg.eigvalsh(J)
    eigvals = np.sort(eigvals)
    
    # Ensure positive
    eigvals = np.maximum(eigvals, 0.5)
    
    return eigvals

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    m = 12
    
    # Generate multiple initializations
    inits = []
    
    # Dyson gas init
    try:
        dyson_roots = _dyson_gas_init(m, repulsion=4.0)
        inits.append(dyson_roots)
    except Exception:
        pass
    
    # Standard Laguerre zeros
    try:
        x_lag, _ = np.polynomial.laguerre.laggauss(m)
        inits.append(x_lag.copy())
    except Exception:
        pass
    
    # Modified Jacobi eigenvalue inits with different parameters
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        for beta in [0.0, 0.5, 1.0, 1.5]:
            try:
                j_roots = _jacobian_eigen_init(m, alpha=alpha, beta=beta)
                inits.append(j_roots)
            except Exception:
                pass
    
    # Geometric sequence
    for ratio in [1.2, 1.3, 1.5, 2.0]:
        try:
            geo_roots = np.array([2.0 * (ratio ** i) for i in range(m)])
            inits.append(geo_roots)
        except Exception:
            pass
    
    # Uniform with different ranges
    for lo, hi in [(1, 30), (2, 50), (3, 70), (5, 100)]:
        inits.append(np.linspace(lo, hi, m))
    
    # Evaluate all inits and keep the best few
    scored_inits = []
    for init_roots in inits:
        init_roots = np.sort(init_roots)
        if len(init_roots) != m or any(r <= 0 for r in init_roots):
            continue
        log_norm, _ = _compute_norm_log_and_grad(init_roots)
        if np.isfinite(log_norm):
            scored_inits.append((log_norm, init_roots))
    
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 5 distinct initializations
    best_inits = []
    seen = set()
    for log_n, roots in scored_inits:
        key = tuple(np.round(roots, 2))
        if key not in seen:
            best_inits.append(roots)
            seen.add(key)
        if len(best_inits) >= 5:
            break
    
    if not best_inits:
        best_inits = [np.linspace(2.0, 50.0, m)]
    
    # Optimization objective in log-space
    def _log_space_objective(log_params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return 1e10, None
        roots = np.exp(log_params)
        roots = np.sort(roots)
        if any(r <= 1e-6 or r > 1e8 for r in roots):
            return 1e10, None
        log_norm, grad = _compute_norm_log_and_grad(roots)
        if not np.isfinite(log_norm):
            return 1e10, None
        # Chain rule for log-space: d/d(log_r_j) = r_j * d/d(r_j)
        # But with sorting, we need to be careful. Use direct gradient.
        grad_log = grad * roots
        return -log_norm, -grad_log
    
    def _log_space_objective_only(log_params):
        val, _ = _log_space_objective(log_params)
        return val
    
    best_roots = best_inits[0].copy()
    best_log_norm, _ = _compute_norm_log_and_grad(best_roots)
    
    # Optimize from each initial point
    for idx, init_roots in enumerate(best_inits):
        if time.time() - start_time > deadline - 30:
            break
        
        log_init = np.log(init_roots)
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                _log_space_objective_only,
                log_init,
                jac=lambda x: _log_space_objective(x)[1],
                method='L-BFGS-B',
                bounds=[(np.log(0.5), np.log(200.0))] * m,
                options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if np.isfinite(result.fun):
                candidate_roots = np.sort(np.exp(result.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
        except Exception:
            pass
        
        # Polish with tighter tolerances if improved
        if time.time() - start_time > deadline - 20:
            break
        
        log_best = np.log(best_roots)
        try:
            result2 = minimize(
                _log_space_objective_only,
                log_best,
                jac=lambda x: _log_space_objective(x)[1],
                method='L-BFGS-B',
                bounds=[(np.log(0.5), np.log(200.0))] * m,
                options={'maxiter': 2000, 'maxfun': 100000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if np.isfinite(result2.fun):
                candidate_roots = np.sort(np.exp(result2.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
        except Exception:
            pass
    
    # Final Nelder-Mead polish if time permits
    if time.time() - start_time < deadline - 15:
        try:
            result3 = minimize(
                _log_space_objective_only,
                np.log(best_roots),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True}
            )
            if np.isfinite(result3.fun):
                candidate_roots = np.sort(np.exp(result3.x))
                cand_log_norm, _ = _compute_norm_log_and_grad(candidate_roots)
                if cand_log_norm > best_log_norm:
                    best_roots = candidate_roots
                    best_log_norm = cand_log_norm
        except Exception:
            pass
    
    # Final verification
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != m:
        best_roots = np.linspace(2.0, 50.0, m)
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
