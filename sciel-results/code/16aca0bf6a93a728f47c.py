import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    # Best Fit: Minimize remaining capacity (Maximize -remaining)
    # We want the highest score for the smallest remaining.
    # So score = -remaining
    scores[valid] = -remaining[valid]
    
    # Tie-breaking: Prefer lower index (First Fit style)
    # Subtract a tiny penalty proportional to index
    # This ensures that if two bins have the exact same remaining capacity,
    # the one with the lower index gets a slightly higher score.
    scores[valid] -= np.arange(len(bins))[valid] * 1e-9
    
    return scores
