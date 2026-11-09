import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for an online bin-packing heuristic.
    
    Uses a Best-Fit strategy: prefer bins with the least remaining capacity
    that can still fit the item. This minimizes fragmentation by filling bins
    as completely as possible, which is beneficial for reducing the number
    of bins used (and thus the excess over the L1 lower bound).
    """
    # Filter bins that can fit the item
    valid_mask = bins >= item
    
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(bins, -1e10, dtype=np.float64)
    
    if np.any(valid_mask):
        # For valid bins, score is the negative of remaining capacity.
        # Maximizing this score selects the bin with the least free space (Best-Fit).
        # We subtract a small constant to avoid zero scores which might be ambiguous.
        scores[valid_mask] = -bins[valid_mask]
        
    return scores
