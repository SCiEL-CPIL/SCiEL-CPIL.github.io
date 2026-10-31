import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Structural change: Switch from Best Fit (minimize remaining capacity)
    # to Worst Fit (maximize remaining capacity).
    #
    # Rationale:
    # Best Fit tends to create many bins with small, unusable remaining capacities
    # (fragments). For Weibull distributions, which can have significant variance
    # and occasional large items, preserving large remaining capacities is crucial
    # to accommodate future large items without opening new bins.
    # Worst Fit keeps the "pool" of available capacity as large as possible,
    # reducing the probability that a future large item will not fit in any existing bin.
    
    # Identify bins that can accommodate the current item
    valid_mask = bins >= item
    
    # Initialize scores to negative infinity so invalid bins are never selected
    scores = np.full(bins.shape, -np.inf, dtype=float)
    
    # For valid bins, the score is the remaining capacity.
    # Since the item size is constant for this decision, maximizing (bins - item)
    # is equivalent to maximizing bins.
    scores[valid_mask] = bins[valid_mask]
    
    return scores
