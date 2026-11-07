"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and UCB-style
    exploration for efficient minimization of Styblinski-Tang 1D.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term: balances exploitation and exploration
    # Use a moderate beta to encourage exploration in early iterations
    beta = 1.5
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB: use EI for exploitation and a portion of sigma for exploration
    # The weight on sigma helps discover new basins in this multimodal function
    score = ei + 0.3 * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: return immediately if we're close to the limit
    elapsed = time.time() - start_time
    if elapsed > 295.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
