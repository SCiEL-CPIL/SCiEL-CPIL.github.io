import numpy as np
import time

# Module-level state for deadline tracking
_start_time = time.time()

# Estimated median item size for Weibull sequences with capacity 100
_MEDIAN_ITEM_SIZE = 4.0
_CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of one score per bin (higher = better).
    
    Strategy: Adaptive hybrid scoring with smooth nonlinear transforms:
    - Best-fit via smooth inverse (1/(rem+eps))
    - Next-item-fit via Gaussian centered at median item size
    - Worst-fit via linear preference for larger remaining (preserves capacity)
    - Small-gap penalty to avoid unusable tiny gaps
    - Content bonus to encourage filling partially-used bins
    """
    # Deadline guard: if we're running long, fall back to simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        scores[mask] = -remaining[mask]
        return scores

    remaining = bins - item
    mask = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(mask):
        return scores
    
    rem = remaining[mask]
    used = bins[mask] - rem  # used capacity in feasible bins
    
    # Component 1: Best-fit via smooth inverse transform
    eps = 1e-6
    best_fit_score = 1.0 / (rem + eps)
    
    # Component 2: Next-item-fit - prefer remaining close to median item size
    # Use a smooth Gaussian-like kernel
    sigma = 3.0
    diff = rem - _MEDIAN_ITEM_SIZE
    next_item_fit_score = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Component 3: Worst-fit component - prefer emptier bins (higher remaining)
    # Normalize to [0, 1]
    worst_fit_score = rem / _CAPACITY
    
    # Component 4: Small-gap penalty - penalize creating tiny unusable gaps
    # For Weibull items with capacity 100, gaps < 2.0 are likely unusable
    small_gap_threshold = 2.0
    # Smooth penalty: quadratic for rem < threshold
    small_gap_penalty = np.where(rem < small_gap_threshold, 
                                  (small_gap_threshold - rem) ** 2 * 0.5, 
                                  0.0)
    
    # Component 5: Content bonus - encourage placing in partially-filled bins
    # This reduces fragmentation by consolidating items
    content_bonus = 0.1 * (used / _CAPACITY)
    
    # Adaptive blending based on item size
    if item > 15.0:
        # Very large items: primarily worst-fit to preserve fragmentation
        w_bf = 0.25
        w_nif = 0.15
        w_wf = 0.50
        w_content = 0.10
    elif item > 8.0:
        # Large items: balance between best-fit and worst-fit
        w_bf = 0.45
        w_nif = 0.20
        w_wf = 0.25
        w_content = 0.10
    elif item > 4.0:
        # Medium items: balance between best-fit and next-item-fit
        w_bf = 0.35
        w_nif = 0.35
        w_wf = 0.20
        w_content = 0.10
    else:
        # Small items: primarily next-item-fit and best-fit
        w_bf = 0.30
        w_nif = 0.40
        w_wf = 0.15
        w_content = 0.15
    
    # Normalize best_fit_score to [0, 1] for comparable scale
    max_bf = np.max(best_fit_score) if len(best_fit_score) > 0 else 1.0
    if max_bf > 0:
        best_fit_score_norm = best_fit_score / max_bf
    else:
        best_fit_score_norm = best_fit_score
    
    # Combine all components
    final_scores = (w_bf * best_fit_score_norm + 
                    w_nif * next_item_fit_score +
                    w_wf * worst_fit_score +
                    w_content * content_bonus) - small_gap_penalty
    
    scores[mask] = final_scores
    
    return scores
