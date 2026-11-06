"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit
    We prefer bins with the largest remaining capacity after placing the item.
    This spreads items across bins, potentially preserving bins with small remaining
    capacities for future small items and reducing fragmentation.
    """
    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Create a mask for bins that can fit the item
    mask = remaining >= 0
    
    # Apply the score: we want the largest remaining capacity to have the highest score.
    # For invalid bins, score is -inf.
    scores = np.full_like(remaining, -np.inf, dtype=float)
    scores[mask] = remaining[mask]
    
    return scores
# EVOLVE-BLOCK-END
