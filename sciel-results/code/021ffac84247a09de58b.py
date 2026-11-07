"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute heuristic values for ant colony optimization.
    
    Args:
        prizes: (5, 100) array of prizes per constraint per item
        weights: (5, 100) array of weights per constraint per item
    
    Returns:
        (100,) array of heuristic values, one per item
    """
    # Total prize per item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)
    
    # Total weight per item (sum across constraints)
    total_weight = np.sum(weights, axis=0)
    
    # Value-to-weight ratio as base heuristic
    value_weight_ratio = total_prize / (total_weight + 1e-9)
    
    # Normalize to [0, 1] range for better scaling
    min_vw = np.min(value_weight_ratio)
    max_vw = np.max(value_weight_ratio)
    if max_vw > min_vw:
        normalized_vw = (value_weight_ratio - min_vw) / (max_vw - min_vw)
    else:
        normalized_vw = np.ones_like(value_weight_ratio) * 0.5
    
    # Also consider items with high absolute prize (not just ratio)
    min_p = np.min(total_prize)
    max_p = np.max(total_prize)
    if max_p > min_p:
        normalized_prize = (total_prize - min_p) / (max_p - min_p)
    else:
        normalized_prize = np.ones_like(total_prize) * 0.5
    
    # Combine: weight the ratio more heavily, but also consider absolute value
    heuristic = 0.7 * normalized_vw + 0.3 * normalized_prize
    
    # Ensure all values are positive and within a reasonable range
    # Scale to [1, 10] to avoid extreme values that could cause numerical issues
    heuristic = 1.0 + 9.0 * heuristic
    
    # Ensure no non-positive values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
