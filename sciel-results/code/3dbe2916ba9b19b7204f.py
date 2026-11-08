"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better. The item goes to the highest-scoring bin.
    
    Strategy: Best-fit with a slight preference for bins that are already
    partially filled (to consolidate usage), but penalize bins that would
    become very full relative to their capacity if this item doesn't fit well.
    
    We use a combination of:
    1. Best-fit: minimize remaining space after placing item (classic)
    2. A small bonus for bins that are already in use (to reduce number of bins opened)
    3. A penalty for bins where the item would leave very little space (to avoid fragmentation)
    """
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=np.float64)
    
    # Determine which bins have enough remaining capacity
    # bins[i] represents the remaining capacity of bin i
    valid_mask = bins >= item
    
    # Create score array initialized to very negative (invalid bins get low score)
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        # No valid bin; return all -inf (the caller should handle this)
        return scores
    
    valid_bins = bins[valid_mask]
    
    # Remaining space after placing item in each valid bin
    remaining = valid_bins - item
    
    # Best-fit score: prefer bins with least remaining space
    # We want to maximize score, so use negative of remaining
    best_fit_score = -remaining
    
    # Additional heuristic: prefer bins that are already partially used
    # If a bin has very high remaining capacity (close to full capacity), 
    # it might be better to keep it open for larger items
    # But since we don't know the total capacity directly, we use relative measures
    
    # Normalize remaining to [0, 1] range for the valid bins
    max_remaining = np.max(valid_bins) if len(valid_bins) > 0 else 1.0
    if max_remaining > 0:
        normalized_remaining = remaining / max_remaining
    else:
        normalized_remaining = np.zeros_like(remaining)
    
    # Combine: primarily best-fit, with a small penalty for leaving lots of space
    # This encourages packing items tightly
    score = best_fit_score - 0.1 * normalized_remaining
    
    # Assign scores back to valid positions
    scores[valid_mask] = score
    
    return scores
# EVOLVE-BLOCK-END
