"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses pure Expected Improvement, which has been measured to outperform
    hybrid EI+UCB combinations for this multi-modal landscape.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    start_time = time.time()
    DEADLINE = start_time + 295.0  # Safety margin before 330s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-12)  # Avoid division by zero
    
    # Clip extreme values for numerical stability
    mu = np.clip(mu, f_best - 50, f_best + 50)
    sigma = np.clip(sigma, 1e-10, 100)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Pure Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Check deadline
    if time.time() > DEADLINE:
        return ei
    
    return ei
# EVOLVE-BLOCK-END
