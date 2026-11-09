import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Filter bins that can fit the item
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # Initialize scores with -inf (invalid)
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
        
    # Strategy: Hybrid
    # If item is large (> 50), use Worst-Fit (prefer most remaining space)
    # If item is small (<= 50), use Best-Fit (prefer least remaining space)
    
    # We need to return scores such that the highest score is chosen.
    # For Best-Fit: Score = -remaining (lowest remaining -> highest score)
    # For Worst-Fit: Score = remaining (highest remaining -> highest score)
    
    # To combine, we can just pick one logic based on item size.
    
    if item > 50:
        # Worst Fit
        # We want the bin with the MAX remaining space.
        # So score = remaining.
        scores[valid_mask] = remaining[valid_mask]
    else:
        # Best Fit
        # We want the bin with the MIN remaining space.
        # So score = -remaining.
        scores[valid_mask] = -remaining[valid_mask]
        
    return scores
