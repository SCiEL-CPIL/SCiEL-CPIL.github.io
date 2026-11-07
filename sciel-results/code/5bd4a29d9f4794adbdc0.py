"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for elapsed time tracking
_start_time = [None]

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) 
    with a time-decaying exploration bonus.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviations
    f_best: float - best observed value so far
    
    Returns: array of shape (2000,) - scores (higher = more promising)
    """
    if _start_time[0] is None:
        _start_time[0] = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Probability of Improvement (PI)
    # P(y < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi_scores = norm.cdf(z)
    
    # 2. Exploration Component
    # We use the normalized sigma as a proxy for uncertainty.
    # This term is bounded in [0, 1] and encourages exploring high-uncertainty regions.
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # 3. Time-Adaptive Weight for Exploration
    # Early on, we want more exploration (higher weight on sigma_norm).
    # Later, we rely more on PI (exploitation).
    elapsed = time.time() - _start_time[0]
    # Decay weight from 0.5 to 0.0 over ~100 seconds (approx 10-15 steps)
    # This allows the algorithm to settle into exploitation mode quickly.
    w_explore = 0.5 * np.exp(-0.05 * elapsed)
    
    # 4. Combine
    # PI is in [0, 1]. sigma_norm is in [0, 1].
    # The sum is in [0, 1 + w_explore].
    # Higher score means more likely to be the next evaluation.
    scores = pi_scores + w_explore * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
