"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score means the bin is preferred.
    The item goes to the highest-scoring bin.
    
    Strategy: Worst-Fit with a nonlinear preference for larger remaining capacity.
    We score bins by their remaining capacity after placing the item, raised to a power
    to emphasize bins that will retain more space. This is a form of Worst-Fit Decreasing
    heuristic adapted for online setting.
    """
    # Deadline guard: if we're taking too long, return a simple fallback
    # (This function is called per-item, so it should be fast; but add safety)
    
    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Only bins with enough remaining capacity are valid
    valid = remaining >= 0
    
    # For valid bins, score by remaining capacity with a power transform.
    # Using a power > 1 emphasizes larger remaining capacities (Worst-Fit behavior).
    # Using power = 1 is pure Worst-Fit (max remaining).
    # We use a moderate power to balance between Worst-Fit and First-Fit-like behavior.
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        # For valid bins, compute score as remaining capacity raised to power
        # Power of 1.5 gives a strong preference for bins with more remaining space
        # but not as extreme as very high powers
        power = 1.5
        scores[valid] = remaining[valid] ** power
    
    return scores
# EVOLVE-BLOCK-END
