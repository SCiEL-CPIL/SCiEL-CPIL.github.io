"""Refined Cohn-Goncalves Laguerre polynomial construction optimizer."""
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.special import roots_laguerre
import time
import math
from numpy.polynomial.polynomial import polyfromroots, polymul

# EVOLVE-BLOCK-START

# Precompute factorials for moment computation
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """
    Compute integral of poly(x) * e^{-x} dx over [0, inf).
    coeffs[k] is coefficient of x^k.
    Result = sum_k coeffs[k] * k!
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
    n_cap = min(n, 80)
    return float(np.dot(coeffs[:n_cap], _FACTS[:n_cap]))

def _build_p_and_p2(roots):
    """
    Given 12 roots, build P(x) = prod (x-x_i)^2 and return (P_coeffs, P2_coeffs).
    Coefficients in ascending power order.
    """
    B = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(B, B)          # degree 24
    P2 = polymul(P, P)         # degree 48
    return P, P2

def _compute_c4(roots, need_grad=False):
    """
    Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2.
    Returns (c4, grad) if need_grad, else just c4.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    
    P, P2 = _build_p_and_p2(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    
    c4 = I2 / (I1 * I1)
    
    if not need_grad:
        return float(c4)
    
    # Compute gradient analytically
    grad = np.zeros_like(roots)
    m = len(roots)
    
    for i in range(m):
        # Build R_i = prod_{j!=i} (x-x_j)^2
        R_i = np.array([1.0])
        for j in range(m):
            if j == i:
                continue
            factor = np.array([roots[j]**2, -2.0*roots[j], 1.0])
            R_i = polymul(R_i, factor)
        
        # (x - x_i) * R_i
        xR_i = np.zeros(len(R_i) + 1)
        xR_i[1:] = R_i
        
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1
        
        # (x - x_i) * R_i * P
        RP = polymul(R_i, P)
        xRP = np.zeros(len(RP) + 1)
        xRP[1:] = RP
        
        term2 = _poly_moment(xRP) - roots[i] * _poly_moment(RP)
        dI2_i = -4.0 * term2
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def _objective(roots):
    """Objective function for optimization."""
    try:
        val = _compute_c4(roots)
        if not np.isfinite(val):
            return 1e100
        return val
    except Exception:
        return 1e100

def _objective_grad(roots):
    """Objective function with gradient."""
    try:
        c4, g = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4):
            return 1e100, np.zeros_like(roots)
        return c4, g
    except Exception:
        return 1e100, np.zeros_like(roots)

def _lognormal_objective(params):
    """
    Objective for log-normal parameterization.
    params = [mu, sigma] where roots are quantiles of lognormal distribution.
    We use the 12 quantiles at positions (i + 0.5)/12 for i=0..11.
    """
    mu, sigma = params
    if sigma <= 1e-6:
        sigma = 1e-6
    # Generate 12 quantiles of lognormal distribution
    # Quantiles at (i+0.5)/12
    from scipy.stats import norm
    quantiles = norm.ppf((np.arange(12) + 0.5) / 12.0)
    roots = np.exp(mu + sigma * quantiles)
    roots = np.sort(roots)
    return _objective(roots)

def _lognormal_objective_grad(params):
    """
    Objective with gradient for log-normal parameterization.
    We need to chain rule: dC4/dmu, dC4/dsigma
    """
    mu, sigma = params
    if sigma <= 1e-6:
        sigma = 1e-6
    
    from scipy.stats import norm
    quantiles = norm.ppf((np.arange(12) + 0.5) / 12.0)
    roots = np.exp(mu + sigma * quantiles)
    roots = np.sort(roots)
    
    c4, grad_roots = _compute_c4(roots, need_grad=True)
    
    # Chain rule: roots_i = exp(mu + sigma * q_i)
    # droot_i/dmu = root_i
    # droot_i/dsigma = root_i * q_i
    # But roots are sorted, so we need to track the permutation
    
    # Actually, since we sort, the gradient gets complicated.
    # For now, use numerical gradient in 2D
    eps = 1e-7
    c4_mu_plus = _lognormal_objective([mu + eps, sigma])
    c4_mu_minus = _lognormal_objective([mu - eps, sigma])
    c4_sigma_plus = _lognormal_objective([mu, sigma + eps])
    c4_sigma_minus = _lognormal_objective([mu, sigma - eps])
    
    grad_mu = (c4_mu_plus - c4_mu_minus) / (2 * eps)
    grad_sigma = (c4_sigma_plus - c4_sigma_minus) / (2 * eps)
    
    if not np.isfinite(c4):
        return 1e100, np.array([0.0, 0.0])
    
    return c4, np.array([grad_mu, grad_sigma])

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Returns: list of 12 positive double-root locations
    """
    start_time = time.time()
    DEADLINE = 630  # seconds
    SAFETY_MARGIN = 45  # seconds
    
    m = 12
    
    def time_ok():
        return (time.time() - start_time) < DEADLINE - SAFETY_MARGIN
    
    # Initial guess: roots of Laguerre polynomial L_12(x)
    try:
        lag_roots, _ = roots_laguerre(m)
        initial_guess = np.sort(lag_roots)
    except Exception:
        initial_guess = 4.0 * np.arange(1, m + 1, dtype=float)
    
    best_x = initial_guess.copy()
    best_val = _compute_c4(best_x)
    
    # Phase 1: Log-normal parameterization with 2D global search
    if time_ok():
        # Try to find good mu, sigma via grid search first
        from scipy.stats import norm
        quantiles = norm.ppf((np.arange(12) + 0.5) / 12.0)
        
        best_ln_val = np.inf
        best_ln_params = None
        
        # Grid search over mu in [0, 4] and sigma in [0.1, 3]
        mu_vals = np.linspace(0.0, 4.0, 17)
        sigma_vals = np.linspace(0.1, 3.0, 15)
        
        for mu in mu_vals:
            for sigma in sigma_vals:
                roots = np.sort(np.exp(mu + sigma * quantiles))
                val = _objective(roots)
                if val < best_ln_val:
                    best_ln_val = val
                    best_ln_params = [mu, sigma]
        
        if best_ln_params is not None and best_ln_val < best_val:
            roots = np.sort(np.exp(best_ln_params[0] + best_ln_params[1] * quantiles))
            best_x = roots.copy()
            best_val = best_ln_val
        
        # Refine with local optimization in 2D
        if time_ok() and best_ln_params is not None:
            try:
                res_ln = minimize(
                    _lognormal_objective,
                    best_ln_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if np.isfinite(res_ln.fun) and res_ln.fun < best_val:
                    mu, sigma = res_ln.x
                    roots = np.sort(np.exp(mu + sigma * quantiles))
                    val = _compute_c4(roots)
                    if val < best_val:
                        best_val = val
                        best_x = roots.copy()
            except Exception:
                pass
    
    # Phase 2: Differential evolution for global search in 12D
    if time_ok():
        bounds = [(1e-4, 100.0)] * m
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=150,
                popsize=12,
                tol=1e-8,
                polish=False,
                update_interval=10,
                init='sobol'
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_val:
                    best_val = c4
                    best_x = candidate.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement with gradient-based optimizer from multiple starts
    candidates_to_refine = [best_x.copy()]
    
    # Add Laguerre roots as another candidate
    candidates_to_refine.append(initial_guess.copy())
    
    # Add perturbed versions
    rng = np.random.default_rng(123)
    for _ in range(3):
        pert = best_x * (1.0 + 0.1 * rng.standard_normal(m))
        pert = np.maximum(pert, 1e-3)
        pert = np.sort(pert)
        candidates_to_refine.append(pert)
    
    # Add evenly spaced and geometric initializations
    candidates_to_refine.append(np.linspace(1.0, 50.0, m))
    candidates_to_refine.append(np.geomspace(1.0, 50.0, m))
    
    # Add log-normal based candidates
    from scipy.stats import norm
    quantiles = norm.ppf((np.arange(12) + 0.5) / 12.0)
    for mu in [1.0, 2.0, 3.0]:
        for sigma in [0.5, 1.0, 1.5]:
            roots = np.sort(np.exp(mu + sigma * quantiles))
            candidates_to_refine.append(roots)
    
    for init in candidates_to_refine:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * m,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_val:
                    best_val = c4
                    best_x = candidate.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish with tighter tolerance
    if time_ok():
        try:
            res_final = minimize(
                _objective_grad,
                best_x,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * m,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            if np.all(res_final.x > 0) and np.isfinite(res_final.fun):
                candidate = np.sort(res_final.x)
                c4 = _compute_c4(candidate)
                if c4 < best_val:
                    best_val = c4
                    best_x = candidate.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_x = np.sort(best_x)
    if np.any(best_x <= 0) or not np.all(np.isfinite(best_x)):
        best_x = initial_guess.copy()
    
    return [float(xi) for xi in best_x]

# EVOLVE-BLOCK-END
