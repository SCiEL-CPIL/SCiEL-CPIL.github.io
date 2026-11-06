"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using the harmonic mean of remaining capacity and item size.
    This favors bins where the item size is a significant portion of the remaining space,
    approximating a balance between filling bins (reducing count) and avoiding
    excessive fragmentation (preserving space for larger items).
    """
    remaining = bins - item
    valid = remaining >= 0
    
    # To avoid division by zero or warnings, use a safe array for calculation
    # where invalid bins are replaced with a dummy value
    safe_remaining = np.where(valid, remaining, 1.0)
    
    # Harmonic Mean H = 2 * R * I / (R + I)
    # We maximize this score
    numerator = 2.0 * safe_remaining * item
    denominator = safe_remaining + item
    
    # Prevent division by zero if item is 0 (though item should be > 0)
    # If item is 0, H is 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = numerator / denominator
        
    # Set invalid bins to -infinity so they are never chosen
    scores = np.where(valid, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
