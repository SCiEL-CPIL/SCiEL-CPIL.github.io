"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Initialize scores to -infinity for invalid bins (cannot fit item)
    scores = np.full_like(rem, -np.inf, dtype=float)
    
    # Identify valid bins where remaining capacity is non-negative
    valid = rem >= 0
    
    if np.any(valid):
        # Best Fit strategy:
        # We want to select the bin with the smallest remaining space that fits the item.
        # Since the evaluator selects the bin with the HIGHEST score,
        # we assign scores such that smaller 'rem' results in a higher score.
        # score = -rem achieves this.
        scores[valid] = -rem[valid]
        
    return scores
# EVOLVE-BLOCK-END
