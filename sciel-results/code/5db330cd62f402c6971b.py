"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
# 330s wall clock limit, keep 10s margin for safety
_DEADLINE = _START_TIME + 320

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PoI) with robust numerical handling.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Fallback: prefer cheap points if we are running out of time
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe

    # Sanitize inputs to prevent numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to a reasonable range to ensure numerical stability of cdf
    # norm.cdf(-10) ~ 7e-25, norm.cdf(10) ~ 1.0
    # Clipping prevents extreme values from causing floating point issues or
    # effectively flattening the score landscape in the tails.
    z = np.clip(z, -10.0, 10.0)
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Cost-aware: PoI per unit cost
    # We want to maximize this
    scores = poi / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
