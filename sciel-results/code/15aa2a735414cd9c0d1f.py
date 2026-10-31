"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    
    Strategy: Hybrid Best/Worst Fit with a penalty for creating very small remainders.
    - Small items: Prefer Best-Fit (minimize remaining space), BUT penalize bins 
      where the remaining space is very small (< 5% of capacity) to avoid 
      creating unusable gaps.
    - Large items: Prefer Worst-Fit (maximize remaining space) to preserve 
      flexibility.
    
    The transition between Best and Worst Fit is smooth based on item size.
    """
    start_time = time.time()
    deadline = 115.0  # 120s limit, 5s safety margin
    
    capacity = 100.0
    
    # Compute remaining capacity if we place the item in each bin
    remaining = bins - item
    
    # Valid bins are those with remaining >= 0
    valid_mask = remaining >= 0
    
    # If no valid bin, return -inf for all
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    # Initialize scores
    scores = np.full_like(bins, -np.inf)
    
    # Item size ratio: how large is the item relative to capacity
    item_ratio = item / capacity  # 0 to 1
    
    # Smooth transition: weight for best-fit vs worst-fit
    # For small items (item_ratio < 0.5), prefer best-fit
    # For large items (item_ratio > 0.5), prefer worst-fit
    # Transition centered at 0.5, width ~0.2
    
    transition_center = 0.5
    transition_width = 0.2
    
    # Smooth step function (sigmoid)
    x = (item_ratio - transition_center) / transition_width
    x = np.clip(x, -5.0, 5.0)
    worst_fit_weight = 1.0 / (1.0 + np.exp(-x))
    best_fit_weight = 1.0 - worst_fit_weight
    
    # Penalty for small remainders in Best-Fit mode
    # We want to avoid creating gaps that are too small to be useful.
    # Threshold: 5.0 units (5% of capacity)
    gap_threshold = 5.0
    
    # Calculate penalty: smooth quadratic penalty for remaining < gap_threshold
    # penalty = (gap_threshold - remaining)^2 if remaining < gap_threshold, else 0
    # Using np.maximum to ensure non-negative difference
    diff = np.maximum(0.0, gap_threshold - remaining)
    penalty = diff * diff
    
    # Base scores
    # Best-Fit score: minimize remaining -> -remaining
    # We subtract the penalty to make bins with small gaps less attractive
    bf_score = -remaining - penalty
    
    # Worst-Fit score: maximize remaining -> +remaining
    wf_score = remaining
    
    # Combine scores
    # score = best_fit_weight * bf_score + worst_fit_weight * wf_score
    scores[valid_mask] = (best_fit_weight * bf_score[valid_mask] + 
                          worst_fit_weight * wf_score[valid_mask])
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores

# EVOLVE-BLOCK-END
