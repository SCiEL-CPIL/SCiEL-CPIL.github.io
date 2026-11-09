"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the bin with the highest score.
    
    Strategy: Hybrid approach combining Best-Fit and Worst-Fit based on item size.
    
    For small items (relative to capacity), use Best-Fit to fill bins tightly.
    For large items, use Worst-Fit to preserve space for other large items.
    
    Additionally, we add a small bonus for bins that are already partially filled
    to encourage consolidation and reduce the number of bins opened.
    
    The bins array represents remaining capacity.
    """
    # Capacity is assumed to be 100 based on the problem statement
    capacity = 100.0
    
    # Remaining capacity after placing the item
    remaining_after = bins - item
    
    # Valid bins are those with enough remaining capacity
    valid = bins >= item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    # For valid bins, compute a hybrid score
    # 
    # Idea: 
    # - If the item is small (< 20% of capacity), prefer Best-Fit (minimize remaining)
    # - If the item is large (> 20% of capacity), prefer Worst-Fit (maximize remaining)
    # - Add a small incentive to use already-active bins (bins that aren't full)
    
    # Determine if item is small or large
    small_item = item < 0.2 * capacity
    
    if small_item:
        # Best-Fit: minimize remaining space after placement
        # Score = -remaining_after (so higher score = less remaining space)
        scores[valid] = -remaining_after[valid]
    else:
        # Worst-Fit: maximize remaining space after placement
        # Score = remaining_after (so higher score = more remaining space)
        scores[valid] = remaining_after[valid]
    
    # Add a small bonus for bins that are already partially used (not empty)
    # A bin is "active" if its load > 0, i.e., remaining < capacity
    active = bins < capacity - 1e-9  # bins that have been used
    scores[valid & active] += 0.01  # small bonus for active bins
    
    return scores
# EVOLVE-BLOCK-END
