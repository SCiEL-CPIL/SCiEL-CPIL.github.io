"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _compute_log_norm_sq(roots, n_quad=200):
    """
    Compute the log of the L2 norm squared of P(x) = prod (x - r_i)^2
    with weight e^{-x}. We want to MAXIMIZE this to MINIMIZE C4.
    """
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 2.0 * np.log(diff)
    
    log_integrand = -x_quad + log_p_sq
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf
    
    exp_shifted = np.exp(log_integrand - max_log)
    sum_val = np.sum(w_quad * exp_shifted)
    if sum_val <= 0:
        return -np.inf
    
    return max_log + np.log(sum_val)

def _compute_log_norm_sq_grad(roots, n_quad=200):
    """Compute log norm squared and its gradient w.r.t. roots."""
    m = len(roots)
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    log_p_sq = np.zeros_like(x_quad)
    for r in roots:
        diff = np.abs(x_quad - r)
        diff[diff < 1e-300] = 1e-300
        log_p_sq += 2.0 * np.log(diff)
    
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
    
    grad = np.zeros(m)
    for j, r in enumerate(roots):
        diff = x_quad - r
        diff[np.abs(diff) < 1e-300] = 1e-300
        grad[j] = 2.0 * np.sum(weighted / diff)
    
    return log_norm_sq, grad

def _get_gauss_laguerre_nodes(alpha, m=12):
    """
    Get the first m nodes of the generalized Gauss-Laguerre quadrature
    for weight w(x) = e^{-x} x^alpha.
    
    Uses the fact that these are the zeros of the generalized Laguerre
    polynomial L_m^{(alpha)}(x).
    """
    from numpy.polynomial.laguerre import laggauss
    # For alpha=0, this is standard Laguerre
    # For general alpha, we need generalized Laguerre polynomial roots
    # Use scipy's roots_laguerre for alpha=0, otherwise compute via companion matrix
    if abs(alpha) < 1e-10:
        x, _ = laggauss(m)
        return np.sort(x)
    
    # Build the generalized Laguerre polynomial L_m^{(alpha)}(x)
    # using its recurrence relation
    # L_0^{(alpha)} = 1
    # L_1^{(alpha)} = 1 + alpha - x
    # (n+1) L_{n+1}^{(alpha)} = (2n + 1 + alpha - x) L_n^{(alpha)} - (n + alpha) L_{n-1}^{(alpha)}
    
    # Build coefficient array for L_m^{(alpha)}(x)
    # Represent polynomial as array of coefficients [c0, c1, ..., cm] where c_k x^k
    L_prev = np.array([1.0])  # L_0
    if m == 0:
        return np.array([0.0])
    L_curr = np.array([1.0 + alpha, -1.0])  # L_1 = (1+alpha) - x
    
    for n in range(1, m):
        # L_{n+1} = ((2n+1+alpha - x) * L_n - (n+alpha) * L_{n-1}) / (n+1)
        # (2n+1+alpha - x) * L_n: multiply L_n by (2n+1+alpha) and subtract x*L_n
        coef = 2*n + 1 + alpha
        # L_n * coef
        term1 = L_curr * coef
        # x * L_n: shift coefficients up by 1
        if len(L_curr) > 0:
            term1 = np.concatenate([np.zeros(1), term1])
            term1[1:] -= L_curr
        # Subtract (n+alpha) * L_{n-1}
        if len(L_prev) > 0:
            term2 = L_prev * (n + alpha)
            # Align lengths
            max_len = max(len(term1), len(term2))
            term1 = np.pad(term1, (0, max_len - len(term1)))
            term2 = np.pad(term2, (0, max_len - len(term2)))
            term1 -= term2
        L_curr = term1 / (n + 1)
        L_prev = L_curr
    
    # Find roots of the polynomial
    # Convert to numpy polynomial form (highest degree first)
    coeffs = L_curr[::-1]
    roots = np.roots(coeffs)
    # Take only real positive roots
    roots_real = roots[np.abs(roots.imag) < 1e-10].real
    roots_pos = roots_real[roots_real > 0]
    roots_pos = np.sort(roots_pos)
    
    if len(roots_pos) < m:
        # Fallback: use standard Laguerre
        x, _ = laggauss(m)
        return np.sort(x)
    
    return roots_pos[:m]

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    
    Strategy: First optimize over alpha in the generalized Gauss-Laguerre
    node parameterization (1-dim), then refine with full 12-dim optimization.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    def time_left():
        return deadline - time.time()
    
    # Phase 1: 1-dimensional search over alpha
    # The generalized Gauss-Laguerre nodes for weight e^{-x} x^alpha
    # give a natural 1-param family of root configurations
    
    best_roots = None
    best_log_norm = -np.inf
    
    # Coarse scan over alpha
    alpha_grid = np.linspace(-2.0, 20.0, 100)
    
    for alpha in alpha_grid:
        if time_left() < 120:
            break
        try:
            roots = _get_gauss_laguerre_nodes(alpha, 12)
            if len(roots) != 12 or any(r <= 0 for r in roots):
                continue
            log_norm = _compute_log_norm_sq(roots, n_quad=200)
            if np.isfinite(log_norm) and log_norm > best_log_norm:
                best_log_norm = log_norm
                best_roots = roots.copy()
        except Exception:
            pass
    
    # Fine scan around best alpha region
    if best_roots is not None:
        # Find which alpha gave the best result
        best_alpha = alpha_grid[np.argmax([
            _compute_log_norm_sq(_get_gauss_laguerre_nodes(a, 12), 100) 
            if len(_get_gauss_laguerre_nodes(a, 12)) == 12 else -np.inf
            for a in alpha_grid
        ])]
        alpha_fine = np.linspace(best_alpha - 2.0, best_alpha + 2.0, 50)
        for alpha in alpha_fine:
            if time_left() < 100:
                break
            try:
                roots = _get_gauss_laguerre_nodes(alpha, 12)
                if len(roots) != 12 or any(r <= 0 for r in roots):
                    continue
                log_norm = _compute_log_norm_sq(roots, n_quad=300)
                if np.isfinite(log_norm) and log_norm > best_log_norm:
                    best_log_norm = log_norm
                    best_roots = roots.copy()
            except Exception:
                pass
    
    # Also try standard Laguerre and a few other initializations
    from scipy.special import roots_laguerre
    x_std, _ = roots_laguerre(12)
    log_norm_std = _compute_log_norm_sq(x_std, n_quad=300)
    if np.isfinite(log_norm_std) and log_norm_std > best_log_norm:
        best_log_norm = log_norm_std
        best_roots = x_std.copy()
    
    if best_roots is None:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        best_log_norm = _compute_log_norm_sq(best_roots, n_quad=300)
    
    # Phase 2: Full 12-dimensional refinement using L-BFGS-B
    def objective_12d(params, n_quad=300):
        if time.time() - start_time > deadline - 30:
            return 1e10
        roots = np.sort(params)
        if any(r <= 0.1 for r in roots):
            return 1e10
        log_norm, _ = _compute_log_norm_sq_grad(roots, n_quad=n_quad)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    def objective_12d_grad(params, n_quad=300):
        if time.time() - start_time > deadline - 30:
            return 1e10, np.zeros(12)
        roots = np.sort(params)
        if any(r <= 0.1 for r in roots):
            return 1e10, np.zeros(12)
        log_norm, grad = _compute_log_norm_sq_grad(roots, n_quad=n_quad)
        if not np.isfinite(log_norm):
            return 1e10, np.zeros(12)
        return -log_norm, -grad
    
    # Refine from the alpha-optimal starting point
    try:
        result = minimize(
            lambda p: objective_12d(p, 300),
            best_roots,
            jac=lambda p: objective_12d_grad(p, 300),
            method='L-BFGS-B',
            bounds=[(0.1, None)] * 12,
            options={'maxiter': 500, 'maxfun': 30000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if np.isfinite(result.fun) and result.fun < -best_log_norm:
            best_roots = np.sort(result.x)
            best_log_norm = -result.fun
    except Exception:
        pass
    
    # Phase 3: Differential Evolution for global exploration if time permits
    if time_left() > 180:
        try:
            def de_objective(params):
                if time.time() - start_time > deadline - 30:
                    return 1e10
                roots = np.sort(params)
                if any(r <= 0.1 for r in roots):
                    return 1e10
                log_norm = _compute_log_norm_sq(roots, n_quad=150)
                if not np.isfinite(log_norm):
                    return 1e10
                return -log_norm
            
            # Bounds for DE: roots should be positive and reasonably spread
            # Typical range for 12 Laguerre-like roots: [0.5, 80]
            de_bounds = [(0.5, 80.0)] * 12
            
            de_result = differential_evolution(
                de_objective,
                de_bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol'
            )
            if de_result.success and np.isfinite(de_result.fun) and de_result.fun < -best_log_norm:
                best_roots = np.sort(de_result.x)
                best_log_norm = -de_result.fun
        except Exception:
            pass
    
    # Phase 4: Final polish with higher quadrature precision
    if time_left() > 60:
        try:
            result = minimize(
                lambda p: objective_12d(p, 500),
                best_roots,
                jac=lambda p: objective_12d_grad(p, 500),
                method='L-BFGS-B',
                bounds=[(0.1, None)] * 12,
                options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.isfinite(result.fun) and result.fun < -best_log_norm:
                best_roots = np.sort(result.x)
                best_log_norm = -result.fun
        except Exception:
            pass
    
    # Final verification
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != 12:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
    
    return best_roots.tolist()
# EVOLVE-BLOCK-END
