"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Return a neutral score if we are running out of time
        # This should rarely happen as the function is vectorized
        return np.zeros_like(mu)

    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # For minimization, we want to maximize EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: divide by cost to prioritize low-cost points
    scores = ei / cost
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
