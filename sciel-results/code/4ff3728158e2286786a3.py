"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 320  # Seconds, leaving 10s buffer for the 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a safe, finite array. 
        # Using zeros is neutral. Using mu-based scores might be better but zeros is safest for validity.
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural Change: Penalize uncertainty to prevent exploitation of high-noise regions
    # This makes the acquisition function more conservative in sparse data regimes.
    # A small penalty factor is used to dampen the exploration term.
    penalty_factor = 0.1
    ei_penalized = ei - penalty_factor * sigma_safe
    
    # Cost-aware score: Higher score = better candidate
    # We want to maximize EI per unit cost.
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei_penalized / cost_safe
    
    # Handle any non-finite values (shouldn't happen with safe denominators, but just in case)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
