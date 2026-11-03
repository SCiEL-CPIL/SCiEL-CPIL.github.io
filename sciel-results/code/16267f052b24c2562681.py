"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize
import time

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound given the double-root locations.
    Uses the 4th moment of the weighted polynomial density.
    """
    roots = np.asarray(roots, dtype=float)
    # Build P(x) = prod (x - r_i)^2
    # Compute moments: int x^k P(x)^2 e^{-x} x^n dx for k=0,4
    # Use Gaussian quadrature with Laguerre nodes
    from numpy.polynomial.laguerre import laggauss
    xq, wq = laggauss(64)
    # P(x)^2 = prod (x - r_i)^4
    p_sq = np.ones_like(xq)
    for r in roots:
        p_sq *= (xq - r) ** 4
    weight = xq ** n * np.exp(-xq)
    int0 = np.sum(wq * p_sq * weight)
    int4 = np.sum(wq * p_sq * weight * xq ** 4)
    if int0 < 1e-30:
        return 1e30
    return int4 / int0

def _roots_from_lognormal(mu, sigma, n_roots=12):
    """Generate n_roots as quantiles of lognormal(mu, sigma^2)."""
    # Use equal-probability quantiles
    probs = (np.arange(1, n_roots + 1)) / (n_roots + 1)
    z = np.zeros(n_roots)
    for i, p in enumerate(probs):
        z[i] = _norm_ppf(p)
    return np.exp(mu + sigma * z)

def _norm_ppf(p):
    """Inverse CDF of standard normal using scipy."""
    from scipy.special import ndtri
    return ndtri(p)

def _objective_lognormal(params, m=12, n=1):
    """Objective: C4 bound for lognormal-parameterized roots."""
    mu, sigma = params
    if sigma < 0:
        return 1e30
    roots = _roots_from_lognormal(mu, sigma)
    if np.any(roots <= 0) or np.any(roots > 1e6):
        return 1e30
    return _compute_c4_bound(roots, m, n)

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Uses lognormal quantile parameterization (2 params) + global search.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    m = 12
    n = 1
    
    best_roots = None
    best_score = 1e30
    
    # Grid search over (mu, sigma)
    mu_range = np.linspace(-2.0, 4.0, 60)
    sigma_range = np.linspace(0.1, 3.0, 40)
    
    for mu in mu_range:
        if time.time() > deadline - 30:
            break
        for sigma in sigma_range:
            if time.time() > deadline - 30:
                break
            try:
                score = _objective_lognormal([mu, sigma], m, n)
                if score < best_score:
                    best_score = score
                    best_roots = _roots_from_lognormal(mu, sigma)
            except Exception:
                pass
    
    # Refine with local optimization from best grid point
    if best_roots is not None:
        # Find best mu, sigma
        best_params = None
        for mu in mu_range:
            for sigma in sigma_range:
                try:
                    s = _objective_lognormal([mu, sigma], m, n)
                    if abs(s - best_score) < 1e-12:
                        best_params = [mu, sigma]
                        break
                except Exception:
                    pass
            if best_params is not None:
                break
        
        if best_params is None:
            # Just use the roots we found
            best_params = [0.0, 1.0]
        
        try:
            res = minimize(
                _objective_lognormal,
                best_params,
                args=(m, n),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_roots = _roots_from_lognormal(res.x[0], res.x[1])
        except Exception:
            pass
    
    # Try a few more random restarts in (mu, sigma) space
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        mu = np.random.uniform(-2.0, 4.0)
        sigma = np.random.uniform(0.1, 3.0)
        try:
            score = _objective_lognormal([mu, sigma], m, n)
            if score < best_score:
                best_score = score
                best_roots = _roots_from_lognormal(mu, sigma)
        except Exception:
            pass
    
    if best_roots is None:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
    
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        result = [4.0 * (i + 1) for i in range(12)]
    return result
# EVOLVE-BLOCK-END
