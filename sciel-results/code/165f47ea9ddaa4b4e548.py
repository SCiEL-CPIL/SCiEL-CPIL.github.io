"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute constants for speed
CAPACITY = 100.0
HALF_CAP = CAPACITY / 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using a Polarization heuristic.
    Favors remainders that are close to 0 (full) or close to Capacity (empty),
    avoiding "half-full" bins which are considered inflexible.
    """
    # Filter bins that can fit the item
    remainders = bins - item
    valid_mask = remainders >= 0
    
    # Initialize scores to -inf (or very low) for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores

    # Calculate polarization score for valid bins
    # Score is higher if remainder is far from the middle (HALF_CAP)
    # (R - HALF_CAP)^2 is max at R=0 and R=CAPACITY
    valid_remainders = remainders[valid_mask]
    
    # Polarization score: square of distance from center
    # We want to maximize this score
    pol_scores = (valid_remainders - HALF_CAP) ** 2
    
    # Add a small Best-Fit component to break ties and encourage filling
    # Best Fit prefers small remainders. 
    # If pol_scores are equal, smaller remainder is better.
    # We can combine: Score = pol_scores - epsilon * remainder
    # Or simply rely on polarization.
    # Let's add a tiny best-fit bias to ensure we don't leave bins unnecessarily empty
    # if polarization scores are similar.
    scores[valid_mask] = pol_scores - 0.01 * valid_remainders

    return scores
# EVOLVE-BLOCK-END
