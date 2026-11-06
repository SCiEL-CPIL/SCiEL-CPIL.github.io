import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing an item. Higher score = better bin.
    Uses rank-based scoring (negative count of smaller remainders) 
    which effectively implements Best-Fit with tie-breaking for 
    larger remaining space.
    """
    # Remaining capacity in each bin
    remaining = bins
    
    # Only consider bins with enough capacity
    valid_mask = remaining >= item
    
    if not np.any(valid_mask):
        return np.full_like(bins, -1e10, dtype=np.float64)
    
    # Sort the remainders to allow efficient rank calculation
    # We use the current state of all bins for the distribution proxy
    sorted_remaining = np.sort(remaining)
    
    # Calculate new remainders for valid bins
    new_remainders = remaining[valid_mask] - item
    
    # Use searchsorted to find the count of bins with remainder < new_remainder
    # side='left' means we count elements strictly less than the value
    # This corresponds to the "negative count of other bins whose remainder is smaller"
    counts = np.searchsorted(sorted_remaining, new_remainders, side='left')
    
    # Score is negative count (higher is better, so we want small counts)
    # Add a small tie-breaker favoring larger new remainders (Worst-Fit bias)
    # This helps in cases where Best-Fit creates many identical small gaps
    scores = np.full_like(bins, -1e10, dtype=np.float64)
    scores[valid_mask] = -counts + 1e-6 * new_remainders
    
    return scores
