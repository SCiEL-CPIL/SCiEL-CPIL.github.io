"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses a Hybrid heuristic:
    - For small items (< 40% capacity): Best Fit (minimize remaining space).
    - For large items (>= 40% capacity): Worst Fit (maximize remaining space).
    
    This aims to prevent fragmentation for large items while packing small items tightly.
    """
    remaining = bins - item
    valid_mask = remaining >= 0
    
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    
    # Capacity is 100 as per task description
    capacity = 100.0
    item_ratio = item / capacity
    
    # Threshold: if item is large, use Worst Fit to preserve capacity for future large items
    # If item is small, use Best Fit to minimize waste
    if item_ratio >= 0.4:
        # Worst Fit: Maximize remaining
        scores[valid_mask] = valid_remaining
    else:
        # Best Fit: Minimize remaining (so score is -remaining)
        scores[valid_mask] = -valid_remaining
        
    return scores

# EVOLVE-BLOCK-END
