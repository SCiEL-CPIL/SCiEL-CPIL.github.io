import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: np.array of shape (5, 100) containing prizes for each constraint/item.
        weights: np.array of shape (5, 100) containing weights for each constraint/item.
        
    Returns:
        np.array of shape (100,) containing attractiveness scores for each item.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Calculate total prize per item (sum across the 5 rows)
    item_values = prizes.sum(axis=0)
    
    # Calculate the maximum weight per item across all constraints
    max_weights = weights.max(axis=0)
    
    # Compute heuristic: Efficiency based on max weight
    # Items with high value and low max weight are most attractive.
    epsilon = 1e-9
    eta = item_values / (max_weights + epsilon)
    
    # Sanitize: Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
