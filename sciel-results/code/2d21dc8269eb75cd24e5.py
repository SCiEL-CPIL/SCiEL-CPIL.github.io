"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and return early if we're close to the limit.
_DEADLINE_SECONDS = 325.0  # 5s safety margin
_start_time = None


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # If we're running out of time, return a simple score to avoid timeout
    if _elapsed() > _DEADLINE_SECONDS:
        return -np.asarray(mu, dtype=np.float64).copy()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Use Expected Improvement (EI)
    # EI is the standard choice for minimization
    z = (f_best - mu) / sigma
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
