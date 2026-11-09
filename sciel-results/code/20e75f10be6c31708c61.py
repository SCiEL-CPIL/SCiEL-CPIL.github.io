"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr, ndtri
import math

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition function that maximizes the predicted 5th 
    percentile of the posterior distribution.
    
    Strategy:
    - The 5th percentile is mu - 1.645 * sigma, which directly targets the 
      lower tail of the predicted distribution.
    - This provides a more robust and aggressive exploitation signal than 
      UCB's additive bonus, particularly effective for unimodal functions 
      like the 1D sphere to rapidly converge to the global minimum.
    - Higher scores indicate points with lower predicted quantiles (better 
      expected outcomes for minimization).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple, fast, safe score if time is critical
        sigma_safe = np.maximum(sigma, 1e-12)
        quantile = mu - 1.645 * sigma_safe
        return np.where(np.isfinite(quantile), quantile, 0.0)

    # Numerical safety for sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute the 5th percentile quantile: mu - 1.645 * sigma
    # This directly targets the lower tail of the posterior
    quantile = mu - 1.645 * sigma_safe
    
    # For minimization, we want to maximize the negative of the quantile
    # (i.e., minimize the quantile itself). Since higher scores mean 
    # "evaluate next", we return -quantile so that lower quantiles get 
    # higher scores.
    score = -quantile
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
