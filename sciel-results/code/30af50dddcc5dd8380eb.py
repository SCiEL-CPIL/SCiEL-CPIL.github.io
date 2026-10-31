"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Strategy: Best-Fit with a penalty for creating very small remainders.
    This avoids fragmenting bins into unusable gaps, preserving larger
    remainders for future items.
    """
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Initialize scores to -inf to mark invalid bins (item doesn't fit)
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Identify bins that can fit the item
    valid_mask = rem >= 0
    
    if not np.any(valid_mask):
        return scores
        
    valid_indices = np.where(valid_mask)[0]
    rem_valid = rem[valid_indices]
    
    # Heuristic parameters
    # Penalize bins where the remaining capacity after placing the item
    # is less than MIN_GAP, as these gaps are likely unusable for future items.
    MIN_GAP = 5.0
    PENALTY = 1000.0
    
    # Base score: Best-Fit logic (prefer smaller remainders)
    # We maximize the score, so we use negative remainder.
    base_scores = -rem_valid
    
    # Apply penalty to bins that would leave a small gap
    small_gap_mask = rem_valid < MIN_GAP
    base_scores[small_gap_mask] -= PENALTY
    
    # Assign scores back to the full array
    scores[valid_indices] = base_scores
    
    return scores
# EVOLVE-BLOCK-END
