import numpy as np
import time

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred (item goes to highest-scoring bin).
    
    Strategy: Smooth rank-based Best-Fit with size-aware blending.
    
    For small/medium items: Use a smooth surrogate of rank-based best-fit.
    Instead of discrete ranks, use a power-law transformation of normalized
    remainders that preserves the ordering but is sensitive to distribution shape.
    
    For large items (> 50% capacity): Use Worst-Fit to avoid creating unusable gaps.
    """
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    valid_rems = remaining[valid_idx]
    
    if item > 0.5 * CAPACITY:
        # Large item: Prefer bins with the most remaining space (Worst Fit)
        # This avoids creating small, unusable gaps
        scores[valid_idx] = valid_rems
    else:
        # Small/Medium item: Smooth rank-based Best-Fit
        # 
        # The current program uses: score = -count(remaining[j] < remaining[i])
        # This is a discrete rank. A smooth surrogate:
        # 
        # Normalize remainders to [0, 1] range
        # Apply a power transformation to sharpen the preference for small gaps
        # 
        # If min_rem == max_rem, all bins are equivalent
        min_rem = np.min(valid_rems)
        max_rem = np.max(valid_rems)
        
        if max_rem - min_rem < 1e-12:
            # All valid bins have the same remaining capacity
            scores[valid_idx] = 0.0
        else:
            # Normalize to [0, 1]
            norm_rems = (valid_rems - min_rem) / (max_rem - min_rem)
            
            # Apply power-law transformation to create a smooth ranking
            # Lower power = more uniform, higher power = sharper preference for small gaps
            # Use power=2 to sharpen the best-fit preference (similar to quadratic penalty)
            # but in a normalized space that handles distribution shape better
            transformed = norm_rems ** 2
            
            # Score: prefer smaller transformed values (smaller gaps)
            # Negate so that smaller gaps get higher scores
            scores[valid_idx] = -transformed
    
    return scores
