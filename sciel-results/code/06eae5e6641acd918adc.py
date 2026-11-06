"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin for 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a Large-Item Preserving Best-Fit heuristic.
    
    Strategy:
    - For small items (relative to capacity), use Best-Fit (minimize remaining space)
      to pack tightly and reduce fragmentation.
    - For large items (relative to capacity), use Worst-Fit (maximize remaining space)
      to preserve smaller bins for medium-sized items that might not fit in the
      leftover space of a large bin if we used Best-Fit.
    
    This hybrid approach aims to capture the benefits of both standard heuristics
    without the unstable transitions of adaptive interpolation.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(bins, dtype=np.float64)

    # Calculate remaining capacity
    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores to a very low value for infeasible bins
    scores = np.full_like(remaining, -1e18, dtype=np.float64)
    
    if not np.any(feasible):
        return scores

    feasible_remaining = remaining[feasible]
    
    # Threshold for switching from Best-Fit to Worst-Fit
    # If the item is larger than 40% of the bin capacity (100), we switch.
    # This prevents putting a large item into a bin where it leaves a gap too small
    # for many other medium items, whereas putting it in a larger bin (if available)
    # or the largest remaining space might be better.
    # Actually, standard Best Fit is usually very strong. 
    # The "Worst Fit" for large items is a common refinement.
    # Let's use a threshold of 0.5 * max_bin_capacity? 
    # The task says capacity 100.
    
    # Heuristic: 
    # If item > 50 (half capacity), use Worst Fit (maximize remaining).
    # Else, use Best Fit (minimize remaining).
    
    # Why?
    # If item is small, filling the tightest hole is efficient.
    # If item is large, putting it in the largest hole preserves the smaller holes
    # which are more versatile for the remaining distribution of items.
    
    threshold = 50.0
    
    if item > threshold:
        # Worst Fit: Score is proportional to remaining space (higher is better)
        scores[feasible] = feasible_remaining
    else:
        # Best Fit: Score is inversely proportional to remaining space (lower remaining is better)
        # We use negative remaining so that the bin with the smallest remaining space gets the highest score.
        scores[feasible] = -feasible_remaining
        
    return scores
# EVOLVE-BLOCK-END
