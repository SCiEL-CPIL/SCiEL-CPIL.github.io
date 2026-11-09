"""Cost-aware acquisition for Hartmann 6D with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 25.0  # seconds of safety margin before 330s kill

def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330.0 - _DEADLINE_MARGIN
    return _DEADLINE

def _time_exceeded():
    return time.time() >= _ensure_deadline()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviation predictions
    f_best: float, lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns: (2000,) array of finite scores (higher = evaluate next)
    """
    _ensure_deadline()
    
    # Clamp sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Cost-adjusted Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Divide by cost to get improvement per unit cost
    # cost = 1 + mean(x) where x in [0,1]^6, so cost in [1, 2]
    cost_safe = np.maximum(cost, 1e-10)
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
