"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function.
    
    Args:
        mu: array of shape (2000,) containing posterior means.
        sigma: array of shape (2000,) containing posterior standard deviations.
        f_best: float, the best observed value so far.
        
    Returns:
        array of shape (2000,) containing acquisition scores.
    """
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        # If we are close to the deadline, return a neutral score 
        # to avoid triggering timeouts or invalid states, though 
        # this is unlikely to be hit given the speed of the operation.
        # Returning zeros is a safe fallback that doesn't crash.
        return np.zeros_like(mu)

    # Compute standardized improvement
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # Since we are minimizing, improvement is when mu < f_best.
    # The probability is Phi((f_best - mu) / sigma)
    scores = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
