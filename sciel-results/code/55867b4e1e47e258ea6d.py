"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses Pure Expected Improvement (EI).
    Based on observed improvements when shifting weight from UCB to EI.
    
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
    # Deadline guard: If we are close to the limit, return a safe default
    # The limit is 330s for the whole run. This function is called 25 times.
    # If we are in the last few seconds, we might want to skip expensive ops, 
    # but EI is cheap. We'll just add a check.
    elapsed = time.time() - _start_time
    if elapsed > 325:  # Safety margin
        # Return a score that prioritizes the current best (exploitation)
        # or just zeros to stop exploration? 
        # Returning zeros might break the "highest score" selection if all are 0.
        # Better to return -mu (UCB with beta=0) to at least pick the best mean.
        return -mu

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
