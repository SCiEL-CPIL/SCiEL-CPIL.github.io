import numpy as np
from scipy.stats import norm
import time

_start_time = None
_TIME_LIMIT = 290.0  # safety margin before 300s kill


def _check_deadline():
    """Return True if we should stop (time budget nearly exhausted)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > _TIME_LIMIT


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement for minimization
    # Higher score means we should evaluate this point next
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure positive sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative EI (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: use cost^alpha with alpha > 1 to penalize expensive points more aggressively
    # This encourages the optimizer to find good improvements in cheap regions first
    alpha = 1.5
    cost_factor = np.power(cost, alpha)
    score = ei / cost_factor
    
    # Add a small exploration bonus for high-uncertainty regions to avoid
    # converging too quickly to a local basin
    # The bonus is scaled so it doesn't dominate the EI term
    exploration_bonus = 0.05 * sigma / cost
    
    score = score + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
