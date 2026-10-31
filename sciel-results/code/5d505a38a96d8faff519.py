import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using a polarization heuristic.
    Prefers bins that will be either very full or very empty after placing the item.
    This avoids leaving bins with moderate remaining capacity, which can lead to fragmentation.
    """
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Initialize scores to -infinity to handle invalid bins
    # We maximize the score, so -inf ensures invalid bins are never picked
    scores = np.full(bins.shape, -np.inf, dtype=float)
    
    # Define valid bins (can fit the item)
    # Use a small epsilon to handle floating point precision issues
    valid_mask = remaining >= -1e-9
    
    if np.any(valid_mask):
        # Get remaining space for valid bins
        r = remaining[valid_mask]
        
        # Polarization score: Maximize distance from half-capacity (50)
        # This encourages filling bins to 0 or keeping them at 100
        # r is the space left. 
        # If r is close to 0, |r-50| is ~50.
        # If r is close to 100, |r-50| is ~50.
        # If r is close to 50, |r-50| is ~0.
        scores[valid_mask] = np.abs(r - 50.0)
        
    return scores
