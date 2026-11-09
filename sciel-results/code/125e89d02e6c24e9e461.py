"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    t0 = time.time()
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Use Probability of Improvement (PI) for aggressive exploration
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This gives high scores to regions with high uncertainty or low mean
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Add a small exploitation term based on how much better the mean is
    # This helps distinguish between equally promising regions
    exploitation = 0.01 * np.maximum(f_best - mu, 0)
    
    score = pi + exploitation
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() - t0 > 290.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
