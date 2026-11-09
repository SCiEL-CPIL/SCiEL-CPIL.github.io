"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Bimodal scoring with a slight Worst-Fit bias to exploit Weibull skew.
    - For small items (relative to capacity), prefer bins with more remaining space (Worst-Fit-like)
      to keep bins open for larger items.
    - For larger items, prefer bins that fit well (Best-Fit-like) to minimize fragmentation.
    - The transition is smooth and tuned for Weibull distributions with capacity 100.
    """
    capacity = 100.0  # Given capacity per bin
    
    # Remaining capacity in each bin
    remaining = bins - item
    
    # Only bins with enough remaining capacity are valid
    valid_mask = remaining >= 0
    
    # If no valid bins, return -inf for all (shouldn't happen in valid packing)
    if not np.any(valid_mask):
        return -np.inf * np.ones_like(bins)
    
    # Normalize remaining capacity to [0, 1]
    # Use capacity as the scale
    norm_remaining = remaining / capacity
    
    # For invalid bins, set score to -inf
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # For valid bins, compute bimodal score
    # Weibull skew: most items are small, few are large
    # Threshold: items < ~30% of capacity favor Worst-Fit, larger favor Best-Fit
    
    # Smooth transition using a sigmoid-like function
    # item_ratio = item / capacity
    item_ratio = item / capacity
    
    # Transition center around 0.3, width 0.1
    transition_center = 0.3
    transition_width = 0.1
    
    # Weight for Worst-Fit (prefer more remaining) vs Best-Fit (prefer less remaining)
    # When item is small, we want to prefer bins with MORE remaining (Worst-Fit)
    # When item is large, we want to prefer bins with LESS remaining (Best-Fit)
    
    # Sigmoid: goes from 0 (Best-Fit) to 1 (Worst-Fit) as item gets smaller
    # Actually, let's think: 
    # - Small item -> want to keep bins open -> prefer bins with more remaining -> higher score for more remaining
    # - Large item -> want to fit tightly -> prefer bins with less remaining -> higher score for less remaining
    
    # Let's define:
    # worst_fit_score = norm_remaining  (higher is better for Worst-Fit)
    # best_fit_score = -norm_remaining  (higher is better for Best-Fit, i.e., less remaining)
    
    # Blend based on item size
    # For small items (item_ratio < transition_center), favor Worst-Fit
    # For large items (item_ratio > transition_center), favor Best-Fit
    
    # Use a smooth blending
    t = (item_ratio - transition_center) / transition_width
    # weight_worst_fit: 1 when t is very negative (small item), 0 when t is very positive (large item)
    # Use sigmoid: 1 / (1 + exp(t))
    weight_worst = 1.0 / (1.0 + np.exp(t))
    weight_best = 1.0 - weight_worst
    
    # Compute scores for valid bins
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 0:
        nr = norm_remaining[valid_idx]
        scores[valid_idx] = weight_worst * nr + weight_best * (-nr)
    
    return scores
# EVOLVE-BLOCK-END
