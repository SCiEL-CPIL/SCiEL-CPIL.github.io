"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Cost-Effective Probability of Improvement (CEPI).
    Maximizes P(f(x) < f_best - epsilon) / cost.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return simple exploitation-based score
        score = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Threshold: we want improvement beyond epsilon
    # Use a dynamically relaxed threshold
    epsilon = 0.05
    
    # z-score for threshold: P(f(x) < f_best - epsilon)
    # f(x) ~ N(mu, sigma^2)
    # P(f(x) < f_best - epsilon) = Phi((f_best - epsilon - mu) / sigma)
    z = (f_best - epsilon - mu) / sigma_safe
    
    # Probability of improvement beyond threshold
    pi = norm.cdf(z)
    
    # Cost-aware: divide by cost
    # cost is in [1, 2] since x in [0,10]^4 scaled to [0,1], mean in [0,1]
    # cost = 1 + mean(x_scaled)
    
    score = pi / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
