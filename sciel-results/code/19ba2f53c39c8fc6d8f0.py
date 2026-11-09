"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # For each item, compute the "slack" in each constraint:
    # capacity is 25, so slack = 25 - weight
    # But we want a heuristic that captures how much "room" each item has
    # and how much value it provides.
    
    # A good heuristic for multi-constraint knapsack is based on
    # the ratio of total prize to the maximum weight across constraints,
    # or better, a composite efficiency that accounts for all constraints.
    
    # Approach: For each item, compute the sum of prizes (total value)
    # and the sum of weights (total resource consumption).
    # Then use a geometric mean or product-based efficiency.
    
    # Actually, let's think about what makes an item "attractive" in ACO:
    # 1. High total prize
    # 2. Low weight in all constraints (especially the binding one)
    
    # A simple and effective heuristic: 
    # For each item, compute prize_sum / max_weight across constraints
    # This rewards items with high value relative to their worst-case resource use.
    
    prize_sum = np.sum(prizes, axis=0)  # shape (100,)
    max_weight = np.max(weights, axis=0)  # shape (100,)
    
    # Use a small epsilon to avoid division by zero
    eps = 1e-9
    heur = prize_sum / (max_weight + eps)
    
    # Ensure values are finite and positive
    heur[~np.isfinite(heur)] = 1e-9
    heur[heur <= 0] = 1e-9
    
    return heur
# EVOLVE-BLOCK-END
