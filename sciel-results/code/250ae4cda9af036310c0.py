"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_CAPACITY = 100.0

# Weibull distribution parameters (shape k=1.5, scale lambda=15)
# E[X] = lambda * Gamma(1 + 1/k)
_WEIBULL_K = 1.5
_WEIBULL_LAMBDA = 15.0
_WEIBULL_MEAN = _WEIBULL_LAMBDA * 0.9027  # Gamma(1+1/1.5) ≈ 0.9027


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Hybrid scoring with expected-next-item adjustment:
    
    - Adjusts remaining capacity by subtracting the expected size of the next Weibull item
    - Smooth transition from Best-Fit (small items) to Worst-Fit (large items)
    - Bins that can't fit the expected next item are penalized, encouraging
      consolidation into bins that can accept consecutive items
    """
    remaining_after = bins - item
    valid_mask = remaining_after >= 0

    scores = np.full_like(bins, -np.inf, dtype=np.float64)

    if not np.any(valid_mask):
        return scores

    valid_remaining = remaining_after[valid_mask]

    # Relative item size
    t = item / _CAPACITY

    # Sigmoid weight: w ≈ 1 for small items (Best-Fit), w ≈ 0 for large (Worst-Fit)
    k = 20.0
    t0 = 0.5
    w = 1.0 / (1.0 + np.exp(k * (t - t0)))

    # Adjust remaining by expected next item size
    # Bins that can't fit the expected next item get penalized
    adjusted_remaining = valid_remaining - _WEIBULL_MEAN

    # score = w * (-adjusted_remaining) + (1-w) * (adjusted_remaining)
    #       = (1 - 2w) * adjusted_remaining
    coeff = 1.0 - 2.0 * w
    scores_valid = coeff * adjusted_remaining

    scores[valid_mask] = scores_valid
    return scores
# EVOLVE-BLOCK-END
