import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Worst-Fit heuristic: score bins by their remaining capacity.
    This forces items into the emptiest bins, preserving high-capacity
    bins for future large items and reducing fragmentation.
    
    We use a smooth scoring that emphasizes larger remaining capacities
    while still being computationally efficient.
    """
    remaining = bins
    valid_mask = remaining >= item
    
    scores = np.full(len(bins), -1e18)
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, score by remaining capacity (worst-fit)
    # Use a power function to emphasize larger gaps
    # Higher power = more aggressive worst-fit behavior
    p = 1.5
    
    valid_remaining = remaining[valid_mask]
    # Normalize to [0, 1] range for stability
    min_val = np.min(valid_remaining)
    max_val = np.max(valid_remaining)
    
    if max_val > min_val:
        normalized = (valid_remaining - min_val) / (max_val - min_val)
    else:
        normalized = np.ones_like(valid_remaining)
    
    # Score: emphasize larger remaining capacities
    scores[valid_mask] = normalized ** p
    
    return scores
