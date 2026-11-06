"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Uses a Next-Item Awareness heuristic:
    1. Estimate the expected size of the next item (S_next) from the bin occupancy distribution.
    2. Score bins based on how well their remaining space matches the predicted next item size,
       minimizing the probability of wasting capacity.
    """
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # 1. Identify valid bins (those with enough capacity for the current item)
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return scores
        
    valid_bins = bins[valid_mask]
    new_rems = valid_bins - item
    
    # 2. Estimate the expected size of the next item (S_next)
    # Use the mean of remaining capacities of used bins as a proxy for typical item size.
    # This reflects the distribution of gaps that future items need to fill.
    used_mask = (bins < 100) & (bins > 0)
    if np.any(used_mask):
        used_rems = bins[used_mask]
        # Use mean instead of median for better estimation of typical item size
        s_next = np.mean(used_rems)
    else:
        # If no bins are partially filled, assume the next item is similar to the current one
        s_next = item
        
    # Ensure s_next is positive to avoid division/logic issues
    s_next = max(s_next, 0.1)
    
    # 3. Score bins based on how well their remaining space matches the predicted next item size
    # We want to minimize the "waste" which is the absolute difference between new_rem and s_next
    # But we also want to prefer smaller new_rems (Best-Fit logic) when new_rem is close to s_next
    
    # Calculate a score that rewards bins where new_rem is close to s_next
    # If new_rem is much larger than s_next, we're wasting space
    # If new_rem is much smaller than s_next, the next item won't fit, creating fragmentation
    # The ideal is new_rem ≈ s_next
    
    # Score = -|new_rem - s_next| + bonus for tight fit
    # But we need to handle the case where new_rem < s_next (next item won't fit)
    # In that case, we penalize more heavily
    
    # Calculate the difference
    diff = new_rems - s_next
    
    # Score: negative absolute difference, with additional penalty for new_rem < s_next
    # When new_rem < s_next, the next item won't fit, so we have wasted capacity
    # When new_rem >= s_next, the next item can fit, which is good
    
    # Use a smooth scoring function:
    # - If new_rem >= s_next: score = -new_rem (prefer smaller new_rem, Best-Fit)
    # - If new_rem < s_next: score = -new_rem - penalty * (s_next - new_rem)
    
    # Let's use a different approach:
    # Score = -(new_rem) - penalty_factor * max(0, s_next - new_rem)^2
    
    # This penalizes bins where the remaining space is too small to fit the next item
    # The penalty is quadratic to strongly discourage fragmentation
    
    penalty_factor = 0.1
    
    valid_scores = -new_rems - penalty_factor * np.maximum(0, s_next - new_rems) ** 2
    
    # 4. Assign scores back to the output array
    scores[valid_mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
