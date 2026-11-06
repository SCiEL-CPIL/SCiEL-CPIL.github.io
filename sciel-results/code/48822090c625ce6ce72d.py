import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Initialize scores
    # Best Fit: minimize remaining capacity (maximize negative remaining)
    scores = -remaining
    
    # Ensure bins that cannot fit the item are not selected
    # Set score to -inf for invalid bins
    invalid_mask = bins < item
    scores[invalid_mask] = -np.inf
    
    return scores
