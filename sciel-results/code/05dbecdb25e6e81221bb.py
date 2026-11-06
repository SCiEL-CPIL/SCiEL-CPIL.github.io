"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Uses a Next-Item Awareness heuristic:
    1. Estimate the typical size of the next item (S_next) from the median 
       remaining capacity of currently used bins.
    2. Prefer bins where the remaining space after placing the current item 
       is either very small (tight fit) or large enough to likely fit the 
       next item (S_next).
    3. Penalize bins that would create a "dead zone" gap (0 < gap < S_next).
    """
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # 1. Identify valid bins (those with enough capacity for the current item)
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return scores
        
    valid_bins = bins[valid_mask]
    new_rems = valid_bins - item
    
    # 2. Estimate the expected size of the next item (S_next)
    # We use the median of the remaining capacities of bins that are partially filled.
    # This reflects the "typical" gap size in the system, which is a proxy for
    # the size of items that are expected to fill those gaps.
    used_mask = (bins < 100) & (bins > 0)
    if np.any(used_mask):
        used_rems = bins[used_mask]
        s_next = np.median(used_rems)
    else:
        # If no bins are partially filled, assume the next item is similar to the current one
        s_next = item
        
    # Ensure s_next is positive to avoid division/logic issues, though we only compare
    s_next = max(s_next, 0.1)
    
    # 3. Classify bins as "Safe" or "Unsafe"
    # Safe: The remaining space (new_rem) is >= s_next, so it can likely fit the next item.
    # Unsafe: The remaining space is < s_next, creating a fragment that is likely wasted.
    is_safe = new_rems >= s_next
    
    # 4. Calculate scores
    # Base score: Prefer smaller remaining space (Best-Fit logic)
    base_scores = -new_rems
    
    # Apply penalty to Unsafe bins to push them below Safe bins
    # The penalty must be large enough to ensure any Safe bin is preferred over any Unsafe bin.
    # Max possible new_rem is 100. So a penalty of 1000 is safe.
    penalty = 1000.0
    
    valid_scores = base_scores.copy()
    valid_scores[~is_safe] -= penalty
    
    # 5. Assign scores back to the output array
    scores[valid_mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
