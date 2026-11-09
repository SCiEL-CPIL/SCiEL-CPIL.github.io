import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Capacity is 100
    capacity = 100.0
    threshold = 0.5 * capacity
    
    if item > threshold:
        # Best-Fit: minimize remaining
        scores = np.where(valid, -remaining, -np.inf)
    else:
        # Worst-Fit: maximize remaining
        scores = np.where(valid, remaining, -np.inf)
        
    return scores
