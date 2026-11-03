"""Refined Cohn-Goncalves Laguerre construction: log-normal quantile parameterization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy import optimize

_DEADLINE = 630.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _laguerre_coeffs(n, alpha=1):
    """Compute coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            from math import lgamma
            log_val = lgamma(n + alpha + 1) - lgamma(n - k + 1) - lgamma(k + alpha + 1)
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_integral(poly):
    """Compute integral_0^inf psi(x)^2 x^1 e^{-x} dx for psi with ascending coeffs."""
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    # weights[k] = (k+1)!
    weights = np.ones(max_k + 1)
    for j in range(1, max_k + 1):
        weights[j] = weights[j - 1] * (j + 1)
    return np.dot(sq, weights)


def _c4_bound(roots, m=12):
    """Upper bound on C4 = 1 / integral."""
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    val = _compute_integral(poly)
    if not np.isfinite(val) or val <= 0:
        return float('inf')
    return 1.0 / val


def _lognormal_quantiles(mu, sigma, n_roots=12):
    """Generate n_roots quantiles of a log-normal distribution with params mu, sigma."""
    if sigma <= 0:
        sigma = 1e-10
    # Quantiles: use equally spaced probabilities
    probs = np.linspace(1.0 / (n_roots + 1), n_roots / (n_roots + 1), n_roots)
    # Log-normal quantile: exp(mu + sigma * z) where z = inverse CDF of standard normal
    from scipy.stats import norm
    z = norm.ppf(probs)
    logs = mu + sigma * z
    roots = np.exp(logs)
    return roots


def _c4_from_lognormal(params, m=12, n_roots=12):
    """Compute C4 bound from log-normal parameters (mu, sigma)."""
    mu, sigma = params
    if sigma < 1e-10:
        sigma = 1e-10
    roots = _lognormal_quantiles(mu, sigma, n_roots)
    if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
        return float('inf')
    return _c4_bound(roots, m)


def construct():
    m = 12
    n_roots = 12
    
    # Seed roots: geometric progression
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    if not np.isfinite(seed_val):
        seed_val = float('inf')
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Coarse grid over (mu, sigma) parameter space
    if _time_left() > 60:
        mus = np.linspace(0.0, 5.0, 25)
        sigmas = np.linspace(0.2, 2.5, 20)
        
        for mu in mus:
            if _time_left() < 30:
                break
            for sigma in sigmas:
                if _time_left() < 15:
                    break
                params = np.array([mu, sigma])
                val = _c4_from_lognormal(params, m, n_roots)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_roots = list(_lognormal_quantiles(mu, sigma, n_roots))
    
    # Phase 2: Local optimization from best grid point and several random starts
    if _time_left() > 30:
        # Objective function for scipy minimize
        def objective(params):
            return _c4_from_lognormal(params, m, n_roots)
        
        # Bounds for scipy
        bounds = [(0.0, 6.0), (0.1, 3.0)]
        
        # Start from best grid point
        logs = np.log(best_roots)
        mu0 = float(np.mean(logs))
        sigma0 = float(np.std(logs))
        x0 = np.array([mu0, sigma0])
        
        try:
            result = optimize.minimize(
                objective, x0, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.fun < best_val and np.all(result.x > 0):
                best_val = result.fun
                best_roots = list(_lognormal_quantiles(result.x[0], result.x[1], n_roots))
        except Exception:
            pass
        
        # Multiple random restarts
        np.random.seed(42)
        for restart in range(8):
            if _time_left() < 20:
                break
            mu_r = np.random.uniform(0.5, 5.0)
            sigma_r = np.random.uniform(0.3, 2.5)
            x0_r = np.array([mu_r, sigma_r])
            
            try:
                result = optimize.minimize(
                    objective, x0_r, method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if result.fun < best_val and np.all(result.x > 0):
                    best_val = result.fun
                    best_roots = list(_lognormal_quantiles(result.x[0], result.x[1], n_roots))
            except Exception:
                pass
    
    # Phase 3: Fine-tune by allowing small perturbations of individual roots
    if _time_left() > 20 and best_val < seed_val:
        # Try direct 12D optimization from the log-normal solution
        def objective_12d(roots_arr):
            if np.any(roots_arr <= 0):
                return float('inf')
            return _c4_bound(roots_arr, m)
        
        x0 = np.array(best_roots, dtype=float)
        bounds_12d = [(0.05, 200.0)] * n_roots
        
        try:
            result = optimize.minimize(
                objective_12d, x0, method='Nelder-Mead',
                bounds=bounds_12d,
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_val and np.all(result.x > 0):
                best_val = result.fun
                best_roots = list(result.x)
        except Exception:
            pass
    
    # Ensure all roots are positive
    if any(r <= 0 for r in best_roots):
        best_roots = seed_roots[:]
    
    return best_roots
# EVOLVE-BLOCK-END
