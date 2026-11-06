# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Implements a First-Fit-Diminishing inspired heuristic (Best Fit Online):
    - We want to consolidate items into bins that are already "small" or "full"
      in terms of capacity, to preserve bins with larger remaining capacity 
      for potentially larger future items.
    - Among valid bins (bins[i] >= item), we prefer the one with the smallest 
      current capacity bins[i].
    - Score is set to -bins[i] for valid bins, so higher score corresponds to 
      smaller bins[i].
    - Invalid bins get a score of -infinity.
    """
    # Identify valid bins where the item fits
    valid = bins >= item
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        # Score is negative of the current bin capacity.
        # Maximizing this score minimizes bins[i] among valid bins.
        scores[valid] = -bins[valid]
        
    return scores
# EVOLVE-BLOCK-END
