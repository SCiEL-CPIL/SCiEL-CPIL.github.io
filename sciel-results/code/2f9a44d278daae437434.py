"""Cost-aware acquisition function for Bayesian optimization of Powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_T0 = [None]
_CALLS = [0]


def _erf(x):
    """Vectorized erf using Taylor + rational approximations (fast, no scipy)."""
    x = np.asarray(x, dtype=np.float64)
    # Use a lookup-table / polynomial approach for speed
    # For |x| <= 2.5 use a polynomial approximation; for larger, use 1 - small
    ax = np.abs(x)
    sign = np.sign(x)
    # Polynomial for erf on [0, 2.5] (degree 9, from Abramowitz & Stegun style fit)
    # We'll use a simple 7-term Taylor-like fit that is accurate to ~1e-7 for |x|<3
    # Actually, let's use the well-known 13-term approximation for erf
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
               - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    # y is the approximation of erf(ax) for ax >= 0
    result = sign * y
    return result


def _norm_cdf(z):
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _norm_pdf(z):
    """Standard normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: cost-EI with adaptive exploration.
    
    Args:
        mu: GP means, shape (2000,)
        sigma: GP std devs, shape (2000,)
        f_best: best observed value (scalar)
        cost: evaluation costs, shape (2000,)
        
    Returns:
        scores, shape (2000,), higher = evaluate next
    """
    # Deadline guard
    if _T0[0] is None:
        _T0[0] = time.time()
    _CALLS[0] += 1
    elapsed = time.time() - _T0[0]
    
    # Safety margin: 330s limit, stop heavy compute at 320s
    if elapsed > 320.0:
        s = np.maximum(sigma, 1e-10)
        c = np.maximum(cost, 1e-10)
        score = s / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    s = np.maximum(sigma, 1e-10)
    c = np.maximum(cost, 1e-10)
    
    # Expected Improvement (minimization): EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / s
    # Clip z to avoid numerical issues
    z = np.clip(z, -50.0, 50.0)
    
    Phi = _norm_cdf(z)
    phi = _norm_pdf(z)
    
    ei = (f_best - mu) * Phi + s * phi
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    ei_cost = ei / c
    
    # Adaptive exploration: use UCB-like term with uncertainty
    # Weight decreases with elapsed time (more exploitation later)
    n_calls = _CALLS[0]
    # Typical budget: ~15-20 evals after initial 5
    # Exploration weight: start high, decay
    t_frac = min(elapsed / 280.0, 1.0)
    expl_weight = 1.0 * (1.0 - 0.7 * t_frac)
    
    # UCB term: sigma / cost
    ucb = s / c
    
    # Combine: cost-EI + exploration
    score = ei_cost + expl_weight * ucb
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
