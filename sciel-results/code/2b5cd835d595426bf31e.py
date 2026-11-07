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
    
    # Adaptive exploration: when f_best is still high, explore more;
    # when f_best is close to 0, exploit more.
    # f_best ranges from ~10 (Ackley max) down to 0 (minimum).
    # Use a smooth sigmoid-like mapping for progress to avoid hard kinks.
    # progress ~ 0 when f_best is large, ~ 1 when f_best is near 0.
    progress = 1.0 / (1.0 + np.exp((f_best - 0.5) / 0.3))
    
    # Interpolate beta: start at 2.0 (more exploration) -> end at 0.5 (more exploitation)
    beta = 2.0 - 1.5 * progress
    
    # Interpolate alpha: start at 0.60 (more UCB) -> end at 0.98 (more EI)
    alpha = 0.60 + 0.38 * progress
    
    # Upper Confidence Bound (UCB) for minimization
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a weighted average
    scores = alpha * ei + (1 - alpha) * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result found with a safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
