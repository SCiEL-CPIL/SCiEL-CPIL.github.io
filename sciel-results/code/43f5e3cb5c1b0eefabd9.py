"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Per-constraint efficiency: prize / weight for each constraint and item
    # Add epsilon to avoid division by zero
    eff = prizes / (weights + 1e-9)
    
    # Structural Change: Use the minimum efficiency across constraints (Bottleneck Efficiency)
    # This prioritizes items that are efficient in the most constraining dimension
    # This is a different modeling family than aggregate sum/max penalties
    h = np.min(eff, axis=0)
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
