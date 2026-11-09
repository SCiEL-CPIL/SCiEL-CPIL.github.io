import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for an online bin-packing heuristic.
    
    Uses a refined nonlinear Best-Fit strategy.
    Instead of a linear penalty on remaining capacity, we use a power-law 
    transformation: score = -(remaining / capacity)^alpha.
    
    With alpha < 1, the scoring function is concave. This means:
    1. It still prefers bins with less remaining space (Best-Fit behavior).
    2. It compresses the difference between bins with large remaining space, 
       making the choice less sensitive to noise in the far tail of the distribution.
    3. It effectively creates a "soft" best-fit that can reduce fragmentation 
       compared to strict linear best-fit, especially when items vary in size.
    
    We use alpha = 0.8 as a balance between strict best-fit (alpha=1) and 
    a more relaxed fit (alpha close to 0).
    """
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    # Using -np.inf ensures invalid bins are never selected
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    rem_valid = remaining[valid_mask]
    
    # Capacity is 100 per the task description
    capacity = 100.0
    
    # Normalize remaining by capacity to make the transform scale-invariant
    # This ensures the exponent works consistently regardless of capacity scale
    rem_norm = rem_valid / capacity
    
    # Nonlinear transform: use a power law with exponent < 1
    # alpha = 0.8: Strong preference for tight fits, but smoother than linear
    alpha = 0.8
    
    # Score = -(remaining/capacity)^alpha
    # Higher score (less negative) = tighter fit preferred
    scores[valid_mask] = -(rem_norm ** alpha)
    
    return scores
