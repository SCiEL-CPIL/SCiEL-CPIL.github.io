"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Deadline guard: return fallback if we're getting close to timeout
    # The main computation should be fast, but let's be safe
    deadline = start_time + 290  # 290s margin from 300s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement (PI)
    z = (f_best - mu) / sigma_safe
    
    # PI: Probability that f(x) < f_best
    # This is a structural change from EI (expected magnitude) to PI (likelihood)
    pi = norm.cdf(z)
    
    # Cost-aware scoring:
    # Using a soft cost penalty (square root) instead of linear division.
    # This reduces the harshness of the cost penalty, allowing exploration
    # in moderately expensive regions if the probability of improvement is high.
    # score = PI / sqrt(cost)
    score = pi / np.sqrt(cost)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Fallback to PI if computation takes too long (though it shouldn't)
        return pi
    
    return score
# EVOLVE-BLOCK-END
