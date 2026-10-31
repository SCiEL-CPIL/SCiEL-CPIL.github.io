# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

def _expected_next_item_size() -> float:
    """
    Estimate the expected size of the next item.
    Using the Mode of the Weibull distribution, which represents the most likely item size.
    Mode = lambda * ((k-1)/k)^(1/k) for k > 1.
    """
    if K_WEIBULL <= 1:
        # For k <= 1, mode is 0 (or undefined/lambda depending on definition), 
        # but Weibull with k<1 has infinite density at 0. 
        # Assuming k=2 standard case.
        return 0.0
    return LAMBDA_WEIBULL * ((K_WEIBULL - 1) / K_WEIBULL) ** (1.0 / K_WEIBULL)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Target Fit.
    Score is negative squared difference between remaining space and the expected (mode) next item size.
    This prioritizes bins where the remaining space is close to the likely size of the next item,
    balancing the risk of the next item not fitting (if remaining is too small) 
    and the waste of space (if remaining is too large).
    """
    expected_next = _expected_next_item_size()
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Score: Negative squared difference from expected next item size
    # This creates a peak at remaining == expected_next
    diff = remaining - expected_next
    scores[valid_mask] = -(diff ** 2)
    
    return scores
# EVOLVE-BLOCK-END
