import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s


def _check_deadline():
    """Return True if we should abort and return a fallback immediately."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Quantile-based (5th percentile) for minimization.
    Scores candidates by their posterior 5th percentile (mu - 1.645 * sigma),
    which provides a robust lower-bound estimate balancing exploration and
    exploitation without arbitrary tuning parameters.
    
    mu, sigma : (2000,) arrays, sigma is the GP posterior std dev
    f_best : lowest observed value so far
    Returns: (2000,) scores, higher = more promising
    """
    # Handle edge cases
    if sigma is None or len(mu) != len(sigma):
        return -mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Check deadline
    if _check_deadline():
        return -mu
    
    # Quantile-based acquisition: 5th percentile of posterior
    # z-score for 5th percentile of standard normal
    z_005 = norm.ppf(0.05)  # approximately -1.645
    
    # Score: higher means more promising (lower expected quantile)
    # We want to minimize, so we score by negative of the quantile
    # Quantile = mu + z_005 * sigma (this is the 5th percentile)
    # Higher score = better, so we negate
    scores = -(mu + z_005 * sigma)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
