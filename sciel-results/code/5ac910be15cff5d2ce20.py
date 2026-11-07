"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Cost-Adjusted Probability of Improvement
    (CA-PI). Maximizes the probability of achieving a meaningful improvement
    over f_best, normalized by evaluation cost to enforce cost efficiency.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use a small epsilon to focus on meaningful improvements
    # epsilon scales with the current best to be adaptive
    # For Shekel 4d, the minimum is around -10.5364
    # Use a relative epsilon based on the scale of f_best
    epsilon = 0.1 * max(abs(f_best), 1.0)
    
    # Probability of Improvement with threshold: P(f(x) < f_best - epsilon)
    # = Phi((f_best - epsilon - mu) / sigma)
    z = (f_best - epsilon - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # cost is in [1, 2] since cost = 1 + mean(x) where x in [0,10] scaled to [0,1]
    # Actually cost = 1 + mean(x_scaled) where x_scaled in [0,1], so cost in [1,2]
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-adjusted POI: higher POI per unit cost is better
    score = poi / cost_safe
    
    # Handle any NaNs or infs
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
