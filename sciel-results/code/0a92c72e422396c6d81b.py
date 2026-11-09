"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Constraint-aware crowding penalty.
    1. For each item, find its binding constraint (constraint where it uses the most capacity).
    2. Base score: prize / max_weight across constraints (efficiency).
    3. Crowding penalty: for each constraint c, compute total prize of items binding to c.
       Items binding to a heavily-loaded constraint get penalized.
    4. Combine and apply power amplification.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle edge cases: ensure 2D
    if prizes.ndim != 2 or weights.ndim != 2:
        return np.ones(prizes.shape[1] if prizes.ndim > 1 else 100)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Weight fractions: how much of each constraint's capacity each item uses
    # Shape: (n_constraints, n_items)
    weight_frac = weights / capacity  # fraction of capacity used
    
    # Max weight fraction per item (binding constraint indicator)
    max_frac = weight_frac.max(axis=0)  # (n_items,)
    
    # Base efficiency: prize / max_weight (higher is better)
    # Use max weight to be conservative
    base_score = prizes.sum(axis=0) / (max_frac + 1e-9)  # (n_items,)
    
    # Identify binding constraint for each item (argmax of weight fraction)
    binding = weight_frac.argmax(axis=0)  # (n_items,)
    
    # For each constraint, compute total prize of items that bind to it
    # This measures how "crowded" each constraint is with high-value items
    crowding = np.zeros(n_constraints)
    for c in range(n_constraints):
        mask = (binding == c)
        if mask.any():
            crowding[c] = prizes[:, mask].sum()
    
    # Normalize crowding to [0, 1] range
    max_crowding = crowding.max() + 1e-9
    norm_crowding = crowding / max_crowding  # (n_constraints,)
    
    # Penalty per item: how crowded is its binding constraint?
    item_crowding = norm_crowding[binding]  # (n_items,)
    
    # Apply crowding penalty: reduce score for items in crowded constraints
    # penalty_factor in [0.3, 1.0] - items in very crowded constraints get lower scores
    penalty_factor = 1.0 - 0.7 * item_crowding  # (n_items,)
    
    # Combined heuristic
    heuristic = base_score * penalty_factor
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Power amplification to sharpen distribution (make top items stand out more)
    # Normalize to [0, 1] first
    h_min = heuristic.min()
    h_max = heuristic.max()
    h_range = h_max - h_min + 1e-9
    h_norm = (heuristic - h_min) / h_range  # [0, 1]
    
    # Apply power to amplify differences
    power = 2.0
    h_amplified = h_norm ** power
    
    # Scale back to a reasonable range, keep relative ordering
    # Map from [0, 1] to [1e-9, 1]
    result = 1e-9 + h_amplified
    
    return result
# EVOLVE-BLOCK-END
