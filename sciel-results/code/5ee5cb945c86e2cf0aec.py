import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -infinity
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    # Identify feasible bins
    feasible = remaining >= 0
    
    # For feasible bins, use Best Fit (minimize remaining space)
    # Score = -remaining
    # This ensures the bin with the smallest remaining space gets the highest score.
    scores[feasible] = -remaining[feasible]
    
    return scores
