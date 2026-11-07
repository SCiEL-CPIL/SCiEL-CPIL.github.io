"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    # Deadline guard: return immediately if we're close to the 330s limit
    t0 = time.time()
    deadline = 330.0 - 2.0  # safety margin
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Fixed exploration constant for UCB
    # Based on inspiration: beta=1.5, alpha=0.7 (weight for EI)
    beta = 1.5
    alpha = 0.7
    
    # Upper Confidence Bound (UCB) for minimization
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a fixed weighted average
    scores = alpha * ei + (1 - alpha) * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result found with a safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
