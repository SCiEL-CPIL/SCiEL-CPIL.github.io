"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins based on a 'dead space' metric.
    Penalizes medium-sized remainders (around half capacity) to avoid fragmentation,
    while favoring tight fits (small remainders) and allowing loose fits (large remainders)
    to be less penalized than in standard Best Fit.
    """
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Initialize scores to negative infinity (invalid bins)
    scores = np.full(len(bins), -np.inf)
    
    # Identify valid bins (enough remaining capacity)
    valid_mask = remaining >= 0
    
    if np.any(valid_mask):
        # Normalize remaining capacity to [0, 1]
        # Capacity is fixed at 100 per task description
        C = 100.0
        x = remaining[valid_mask] / C
        
        # U-shaped penalty: minimize distance from center 0.5
        # Score is high when x is near 0 or 1, low when x is near 0.5
        # This avoids creating medium-sized gaps that are hard to fill
        scores[valid_mask] = (x - 0.5) ** 2
        
    return scores
# EVOLVE-BLOCK-END
