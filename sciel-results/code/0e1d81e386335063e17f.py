"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using Best Fit heuristic.
    
    The item is placed in the bin with the smallest remaining capacity that still
    fits the item. This minimizes fragmentation and tends to keep bins full,
    reducing the total number of bins required.
    """
    start_time = time.time()
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Best Fit: Prefer the bin with the smallest remaining space.
    # We assign a score of -remaining, so smaller remaining yields a higher (less negative) score.
    scores[valid_mask] = -remaining[valid_mask]
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
