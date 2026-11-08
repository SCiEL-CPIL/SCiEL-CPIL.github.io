"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time

# Track elapsed time for deadline safety
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _ei_value(mu, sigma, f_best):
    """Compute expected improvement value (not divided by cost)."""
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _gp_posterior(x, mu, sigma):
    """
    Given a candidate point x (8-dim in [-1,1]), compute the GP posterior
    mean and std at x using the provided mu/sigma arrays (which are GP
    predictions at the 2000 candidate points).
    
    Since we don't have the GP kernel parameters or training data directly,
    we use the provided mu and sigma as the GP predictions. For local search,
    we need to approximate how mu and sigma change as x moves. We'll use
    a simple approach: the mu and sigma arrays are fixed (they're the GP
    predictions at the 2000 candidate points), so we can't easily do local
    search on them directly.
    
    Instead, we'll use a different structural approach: combine EI with a
    diversity/exploration term based on distance to already-evaluated regions,
    and use the cost structure more effectively.
    """
    pass

def acquisition(mu, sigma, f_best, cost):
    # Check deadline safety
    elapsed = _get_elapsed()
    if elapsed > 320:
        # Return a simple fallback to avoid timeout
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost

    n = len(mu)
    
    # Compute base EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: use sqrt of cost instead of linear for better exploration
    cost_safe = np.maximum(cost, 1e-6)
    cost_factor = np.sqrt(cost_safe)
    
    # Base score: EI / sqrt(cost)
    score = ei / cost_factor
    
    # Add exploration bonus for high-uncertainty regions
    # This helps find the global minimum in shifted instances
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        # Uncertainty bonus: reward points with high sigma relative to f_best gap
        uncertainty_bonus = 0.1 * sigma / (np.std(sigma) + 1e-9)
        score += uncertainty_bonus
    
    # Add a diversity component: reward points far from the current best region
    # We can approximate this by looking at the distribution of mu values
    # Points with mu much lower than average are likely in good regions
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + 1e-9
    # Reward points that are in the lower tail of mu (likely good regions)
    diversity_bonus = 0.05 * np.exp(-((mu - mu_mean) / mu_std) ** 2)
    score += diversity_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
