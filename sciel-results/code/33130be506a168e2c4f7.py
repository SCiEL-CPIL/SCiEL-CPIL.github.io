"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a bottleneck-slack heuristic.
    
    For each item, compute the minimum of (capacity / weight) across all constraints,
    which measures how many times the item fits in its tightest constraint.
    Multiply by total prize to prioritize high-value items that fit easily.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight values for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger is more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    eps = 1e-9
    
    # Total prize per item
    total_prizes = prizes.sum(axis=0)
    
    # Safe weights to avoid division by zero
    w_safe = np.maximum(weights, eps)
    
    # Capacity over weight for each constraint/item
    cap_over_w = capacity / w_safe  # shape (5, 100)
    
    # Bottleneck: minimum capacity/weight across constraints
    bottleneck = np.min(cap_over_w, axis=0)  # shape (100,)
    
    # Heuristic: total prize * bottleneck
    heuristic = total_prizes * bottleneck
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
