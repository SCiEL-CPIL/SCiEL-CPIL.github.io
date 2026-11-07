"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses Probability of Improvement (PI) acquisition function.
    PI favors points with a high probability of surpassing the current best,
    providing a distinct exploration bias compared to EI.
    
    Parameters:
        mu: array of shape (2000,) - GP mean at candidates
        sigma: array of shape (2000,) - GP std dev at candidates
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: Return a simple score based on mean if we are running out of time
        # This ensures we return a valid array quickly
        return -mu

    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
