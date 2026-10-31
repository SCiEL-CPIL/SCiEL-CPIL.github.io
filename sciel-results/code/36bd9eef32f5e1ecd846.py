# EVOLVE-BLOCK-START
import numpy as np

# Global state to estimate item size distribution
_item_history = []
_MAX_HISTORY = 100

def _estimate_mean(history):
    if not history:
        return 10.0 # Fallback
    return np.mean(history)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Update history
    global _item_history
    _item_history.append(item)
    if len(_item_history) > _MAX_HISTORY:
        _item_history.pop(0)
    
    # Estimate the typical item size
    # Using mean is simple, median might be more robust to outliers
    est_size = _estimate_mean(_item_history)
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Heuristic: Prefer bins where the remaining space is close to the estimated next item size.
    # This balances "Best Fit" (tight) and "Worst Fit" (flexible).
    # Quadratic penalty: score = -(remaining - est_size)^2
    # This maximizes the score when remaining == est_size.
    scores[valid_mask] = -(remaining - est_size) ** 2
    
    return scores
# EVOLVE-BLOCK-END
