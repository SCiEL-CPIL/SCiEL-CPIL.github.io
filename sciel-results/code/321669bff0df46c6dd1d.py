"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time to ensure we return before 330s
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using probability of improvement
    (lower-bound surrogate) normalized by evaluation cost.
    
    This maximizes P(f(x) < f_best - epsilon) / cost, which focuses
    search on high-confidence improvements over a fixed threshold.
    """
    # Check deadline - if close to limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 325.0:  # Safety margin before 330s kill
        # Return simple score based on mu (lower is better) normalized by cost
        cost_safe = np.maximum(cost, 1e-8)
        fallback_scores = -mu / cost_safe
        return np.where(np.isfinite(fallback_scores), fallback_scores, 0.0)
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Adaptive epsilon: use a fraction of the median uncertainty
    # This balances exploration and exploitation
    median_sigma = np.median(sigma_safe)
    alpha = 0.2  # Exploration parameter
    epsilon = alpha * median_sigma
    
    # Target value: we want to find points better than f_best - epsilon
    target = f_best - epsilon
    
    # Calculate z-score for the improvement threshold
    z = (target - mu) / sigma_safe
    
    # Clip z to prevent numerical issues
    z = np.clip(z, -10, 10)
    
    # Probability of improvement: P(f(x) < target) = Phi(z)
    prob_improve = norm.cdf(z)
    
    # The probability is already in [0, 1], higher is better
    # Normalize by cost to get cost-aware score
    scores = prob_improve / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
