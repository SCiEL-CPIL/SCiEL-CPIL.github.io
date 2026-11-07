"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 295.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're getting close to the 300s limit, return safe scores
    if time.time() - _start_time > _TIMEOUT:
        return np.ones_like(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle any remaining numerical issues
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # UCB term: balance exploration and exploitation
    # beta controls exploration; 2.0 was found to work well for Shekel 4d
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost to account for expensive evaluations
    cost_safe = np.maximum(cost, 1e-9)
    
    # Hybrid acquisition: combine EI and UCB, both cost-aware
    # Use a weighted sum where EI dominates exploitation and UCB helps exploration
    ei_score = ei / cost_safe
    ucb_score = ucb / cost_safe
    
    # Combine with a blending factor; 0.5 gives equal weight
    # This helps navigate multimodal landscapes better than pure EI/cost
    blend = 0.5
    score = blend * ei_score + (1 - blend) * ucb_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
