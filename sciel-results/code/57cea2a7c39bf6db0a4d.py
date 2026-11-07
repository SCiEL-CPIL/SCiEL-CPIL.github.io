"""Direct cost-aware acquisition for Bayesian optimization on griewank 2d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: Cost-Adjusted Lower Confidence Bound (LCB).
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions in objective units
    - sigma: array of shape (2000,) - GP std-dev predictions in objective units
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs (1 + mean of scaled coords)
    
    Returns:
    - scores: array of shape (2000,) of finite scores, higher = more promising
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Use Probability of Improvement (PI) weighted by inverse cost.
    # PI = Phi((f_best - mu) / sigma)
    # This strictly prioritizes exploitation of low-mean regions with high confidence.
    
    # Calculate z-scores
    # Avoid division by zero or near-zero sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate PI using the CDF of the standard normal distribution
    pi = norm.cdf(z)
    
    # Weight by inverse cost
    # Score = PI / cost
    scores = pi / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
