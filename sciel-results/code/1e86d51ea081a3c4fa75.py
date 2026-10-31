import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin based on a polarization strategy.
    Bins with remainders close to 0 (full) or close to Capacity (empty) are preferred.
    Bins with remainders close to Capacity/2 are penalized.
    """
    # Capacity is specified as 100 in the task description
    C = 100.0
    
    # Calculate remainders
    remainders = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Identify valid bins
    valid_mask = remainders >= 0
    
    if np.any(valid_mask):
        valid_remainders = remainders[valid_mask]
        
        # Polarization Score: Maximize distance from C/2
        # Using squared distance for stronger contrast
        center = C / 2.0
        scores[valid_mask] = (valid_remainders - center) ** 2
        
    return scores
