# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Implements a First-Fit-Diminishing inspired heuristic:
    - Score is the negative of the current remaining capacity.
    - This forces the selection of the bin with the least available space 
      among those that can fit the item (Best Fit).
    - This aggressively consolidates items, keeping other bins empty 
      for larger future arrivals.
    """
    # Identify valid bins (those with enough capacity)
    valid = bins >= item
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        # Score = -current remaining capacity
        # Maximizing this score is equivalent to minimizing remaining capacity.
        # This selects the bin that is "fullest" but still fits the item.
        scores[valid] = -bins[valid]
        
    return scores
# EVOLVE-BLOCK-END
