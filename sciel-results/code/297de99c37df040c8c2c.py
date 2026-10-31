import numpy as np
import time

# Global state to track item statistics for adaptive strategy
_global_state = {
    "count": 0,
    "sum_items": 0.0,
    "start_time": time.time()
}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Adaptive Best/Worst Fit heuristic.
    
    Strategy:
    - Maintain a running average of item sizes.
    - If the current item is smaller than the average, it suggests that larger items
      might be coming (or the distribution is right-skewed, which Weibull is).
      In this case, we prefer bins with MORE remaining space (Worst Fit) to keep
      space available for those larger items.
    - If the current item is larger than the average, we prefer bins with LESS
      remaining space (Best Fit) to consolidate small spaces and avoid fragmentation.
    
    This adapts the packing strategy dynamically to the observed item distribution.
    """
    # Deadline guard
    if time.time() - _global_state["start_time"] > 110:
        # Fallback to simple Best Fit if running out of time
        remaining = bins - item
        valid_mask = remaining >= 0
        scores = np.full(bins.shape, -np.inf, dtype=float)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores

    # Update global statistics
    _global_state["count"] += 1
    _global_state["sum_items"] += item
    
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores
    scores = np.full(bins.shape, -np.inf, dtype=float)
    valid_mask = remaining >= -1e-9
    
    if not np.any(valid_mask):
        return scores
    
    r = remaining[valid_mask]
    
    # Estimate mean item size
    if _global_state["count"] > 0:
        mean_item = _global_state["sum_items"] / _global_state["count"]
    else:
        mean_item = 50.0  # Default guess
    
    # Calculate bias
    # If item < mean, bias is positive -> we want to maximize r (Worst Fit)
    # If item > mean, bias is negative -> we want to minimize r (Best Fit)
    # We use a scaling factor to control the transition sharpness
    # tanh ensures smooth transition and bounded bias
    diff = mean_item - item
    # Scale diff to a reasonable range for tanh
    # Weibull items with cap 100 typically have mean around 20-40
    # Let's use a scale of 10
    bias = np.tanh(diff / 10.0)
    
    # Score = bias * r
    # If bias > 0, larger r gets higher score (Worst Fit)
    # If bias < 0, smaller r gets higher score (Best Fit, since -r is maximized when r is small)
    # Wait:
    # If bias = 1 (Small item), Score = r. Max score is max r. (Worst Fit)
    # If bias = -1 (Large item), Score = -r. Max score is min r. (Best Fit)
    # This is correct.
    
    scores[valid_mask] = bias * r
    
    return scores
