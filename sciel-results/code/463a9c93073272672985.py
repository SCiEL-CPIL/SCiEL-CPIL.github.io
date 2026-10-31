# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters assumption
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

# Pre-calculate moments
def _get_weibull_moments():
    mean = LAMBDA_WEIBULL * gamma(1 + 1.0 / K_WEIBULL)
    var = LAMBDA_WEIBULL**2 * (gamma(1 + 2.0 / K_WEIBULL) - gamma(1 + 1.0 / K_WEIBULL)**2)
    std = np.sqrt(var)
    return mean, std

_MEAN, _STD = _get_weibull_moments()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Maximize a Gaussian-like score centered at the expected next item size.
    This prioritizes bins where the remaining space is close to the expected size of the next item,
    balancing between packing tightly (low remaining) and keeping space for future items (high remaining).
    """
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return scores
    
    remaining = bins[valid_mask] - item
    
    # Gaussian score: peaks at remaining == _MEAN
    # Width is determined by _STD
    # We use a negative squared distance to make it a "score" where higher is better
    diff = remaining - _MEAN
    scores[valid_mask] = -np.exp(-0.5 * (diff / _STD)**2)
    
    return scores
# EVOLVE-BLOCK-END
