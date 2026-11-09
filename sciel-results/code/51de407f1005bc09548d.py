"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Use squared prize-to-max-weight ratio to sharpen the distribution.
    - Use max weight across constraints (binding constraint) instead of sum
    - Square the ratio to disproportionately favor high-ratio items
    - This helps ants converge faster on high-value, low-impact items
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Use max weight across constraints (the binding constraint)
    max_weight = weights.max(axis=0)
    
    # Base ratio: prize / max_weight
    ratio = prizes / (max_weight + 1e-9)
    
    # Square to sharpen distribution - high ratio items become much more attractive
    scores = ratio ** 2
    
    # Ensure positive values
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
