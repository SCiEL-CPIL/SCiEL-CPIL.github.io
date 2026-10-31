"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit.
    We choose the bin with the maximum remaining space after placing the item.
    This preserves bins with less space for smaller future items, potentially
    reducing fragmentation and allowing larger items to fit into bins that
    would otherwise be too tight.
    
    Score = remaining space (bins - item).
    Higher score means more space left, which is preferred by Worst Fit.
    """
    remaining = bins - item
    
    # Create a score array, initialize to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # For feasible bins (remaining >= 0), compute score
    feasible = remaining >= 0
    if np.any(feasible):
        # Worst Fit: Maximize remaining space
        scores[feasible] = remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
