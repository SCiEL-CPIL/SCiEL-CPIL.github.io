"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds of safety margin

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a hybrid approach combining
    Expected Improvement (EI) with a UCB-like exploration term.
    
    This structural change combines the exploitation strength of EI with
    the exploration benefits of UCB, which has been shown to be more effective
    for multimodal functions like Goldstein-Price.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB score as fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 1.5 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # UCB component: -mu + beta * sigma
    # Using beta=1.5 which balances exploration and exploitation
    beta = 1.5
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB with equal weight
    # Normalize both to have similar scales
    # EI is already in objective units, UCB is also in objective units
    # But we want to emphasize exploration more for multimodal functions
    scores = 0.5 * ei + 0.5 * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
