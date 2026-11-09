<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    return prizes / (weights.sum(axis=0) + 1e-9)
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) with uniform values in [0, 1]
        weights: array of shape (5, 100) with uniform values in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Validate shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("prizes and weights must have shape (5, 100)")
    
    # Check for non-finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("prizes and weights must contain only finite values")
    
    # Each constraint has capacity 25
    capacity = 25.0
    
    # For each item, compute how much of the capacity it would use across all constraints
    # An item j uses weights[:, j] for each of the 5 constraints
    # The "cost" of an item is determined by which constraint it most tightly binds
    
    # Strategy: reward items with high prizes and low weights
    # Use a weighted combination that considers the tightest constraint
    
    # For each item, find the maximum weight across constraints (the binding constraint)
    max_weight_per_item = np.max(weights, axis=0)  # shape (100,)
    
    # Total weight sum across all constraints for each item
    total_weight = np.sum(weights, axis=0)  # shape (100,)
    
    # Prize per unit of total weight (efficiency)
    prize_efficiency = prizes.sum(axis=0) / (total_weight + 1e-9)
    
    # Prize per unit of max weight (how good is it relative to its binding constraint)
    prize_per_max = prizes.sum(axis=0) / (max_weight_per_item + 1e-9)
    
    # Items that fit easily (low max weight relative to capacity) are more attractive
    fit_score = 1.0 / (max_weight_per_item / capacity + 1e-9)
    
    # Combine: we want high prize efficiency and good fit
    # The heuristic should guide the ant colony to pick items that lead to high total prize
    # while respecting all constraints
    
    # A good heuristic: prize / (weight in the most constrained dimension)
    # This favors items that give high prize for low cost in the tightest constraint
    
    # Normalize prizes to [0, 1] range for better scaling
    prize_sum = prizes.sum(axis=0)  # total prize per item
    
    # Weighted combination
    # Emphasize items with high prize-to-weight ratio
    # Also penalize items that are heavy in any single constraint
    
    # Use geometric mean of efficiency and fit
    heuristic = np.sqrt(prize_efficiency * fit_score)
    
    # Add a small component for absolute prize value
    heuristic = 0.7 * heuristic + 0.3 * prize_per_max
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
