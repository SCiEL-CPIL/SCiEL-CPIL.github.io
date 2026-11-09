"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds total wall-clock
_SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimization.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std in objective units
        f_best: float - best (lowest) observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: just use negative mu (exploit best known)
        return -mu.copy()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # We want to maximize EI.
    
    z = (f_best - mu) / sigma
    improvement = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(improvement), improvement, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
