"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Shekel 4d.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: track elapsed time to avoid timeout
    t0 = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: sigma can be very small
    # Use a small epsilon to avoid division by zero in z-score calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamically relaxed threshold
    # We want to find points that are likely to beat f_best by at least 0.05
    # This focuses the search on "meaningful" gains rather than tiny improvements
    threshold = f_best - 0.05
    
    # Calculate z-score for the probability of improvement over the relaxed threshold
    # P(f(x) < threshold) = P((f(x) - mu) / sigma < (threshold - mu) / sigma)
    # Since f(x) ~ N(mu, sigma^2), let Z = (f(x) - mu) / sigma ~ N(0, 1)
    # We want P(f(x) < threshold) = P(Z < (threshold - mu) / sigma) = Phi((threshold - mu) / sigma)
    
    z = (threshold - mu) / sigma_safe
    
    # Probability of Improvement (PI) over the relaxed threshold
    pi = norm.cdf(z)
    
    # Clip PI to be non-negative and finite (CDF is always between 0 and 1, but safety check)
    pi = np.clip(pi, 0.0, 1.0)
    
    # Cost-aware: normalize by cost
    # Use a smoothed cost to avoid division by near-zero
    # Cost is defined as 1 + mean(x scaled to [0, 1]), so it's in [1, 2]
    cost_safe = np.maximum(cost, 1.0)
    
    # Cost-Effective Probability of Improvement (CEPI)
    scores = pi / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time budget
    elapsed = time.time() - t0
    if elapsed > 290:  # Safety margin before 300s limit
        pass  # We're already past most of the computation
    
    return scores
# EVOLVE-BLOCK-END
