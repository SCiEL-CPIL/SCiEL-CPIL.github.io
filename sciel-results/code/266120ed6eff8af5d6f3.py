import numpy as np

# Weibull parameters for item size distribution (shape, scale)
# Typical for bin packing: shape ~1.5, scale ~100
_WEIBULL_SHAPE = 1.5
_WEIBULL_SCALE = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using Weibull hazard rate at the resulting remainder.
    
    The hazard rate h(r) for a Weibull distribution is h(r) = (k/λ)(r/λ)^(k-1).
    By scoring with -h(r), we prioritize remainders where the hazard is low,
    meaning the remainder is unlikely to be immediately utilized by the next item.
    This preserves high-hazard (medium) remainders for items that fit tightly,
    and encourages creating either very small or very large remainders.
    
    This is a structural change from polarization: instead of a fixed midpoint,
    the score is derived from the actual item size distribution's hazard function.
    """
    C = 100.0
    
    # Initialize scores to -inf so invalid bins are never chosen
    scores = np.full(bins.shape, -np.inf)
    
    # Identify bins that can fit the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    # Calculate remaining capacity if the item is placed in the valid bins
    remaining = bins[valid_mask] - item
    
    # Weibull hazard rate: h(r) = (k/λ) * (r/λ)^(k-1)
    # Score = -h(r), so we prefer lower hazard (extreme remainders)
    k = _WEIBULL_SHAPE
    lam = _WEIBULL_SCALE
    
    # Avoid issues with r=0: h(0) = 0 for k>1, which is fine
    # For k<1, h(0) would be infinite, but we use k=1.5 so it's safe
    ratio = remaining / lam
    # Clip to avoid negative values (shouldn't happen since remaining >= 0)
    ratio = np.maximum(ratio, 0.0)
    
    # h(r) = (k/λ) * r^(k-1) / λ^(k-1) = (k/λ^k) * r^(k-1)
    # But simpler: h(r) = (k/λ) * (r/λ)^(k-1)
    hazard = (k / lam) * np.power(ratio, k - 1)
    
    # Score is negative hazard: we want to maximize this (minimize hazard)
    scores[valid_mask] = -hazard
    
    return scores
