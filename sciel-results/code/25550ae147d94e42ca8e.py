"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 320.0  # seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()
    
    # Expected improvement per unit cost
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of improvement (exploration signal)
    poi = norm.cdf(z)
    
    # Cost-aware: divide by cost to prefer cheap evaluations
    cost_safe = np.maximum(cost, 1e-6)
    
    # Weighted combination: more weight on EI for exploitation,
    # but keep POI to encourage exploration in uncertain regions
    # Use a slightly higher exploitation weight (0.7) to focus on
    # promising areas while still exploring
    balance = 0.7
    scores = (balance * ei + (1 - balance) * poi) / cost_safe
    
    # Add a small UCB-like term to encourage exploration of high-uncertainty regions
    # This helps escape local minima in the acquisition landscape
    ucb_term = 0.1 * sigma / cost_safe
    scores = scores + ucb_term
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t_start > _DEADLINE:
        return np.zeros_like(scores)
    
    return scores
# EVOLVE-BLOCK-END
