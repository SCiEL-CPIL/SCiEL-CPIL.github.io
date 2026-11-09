import numpy as np

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    
    The heuristic is based on the ratio of the item's total prize to its maximum
    weight across all constraints. This favors items that are high-value and 
    "balanced" (low weight in their most restrictive dimension), allowing for 
    better packing efficiency in multi-constraint scenarios.
    
    Parameters:
    prizes: array-like, shape (C, N) or (N,)
    weights: array-like, shape (C, N)
    
    Returns:
    np.array: Heuristic scores of shape (N,)
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape variations for prizes
    if prizes.ndim == 2:
        # Assume shape is (Constraints, Items) based on task description
        # Sum prizes across constraints to get total prize per item
        total_prizes = prizes.sum(axis=0)
    else:
        # Assume shape is (Items,)
        total_prizes = prizes
        
    # Handle potential shape variations for weights
    # Task specifies (5, 100), i.e., (Constraints, Items)
    if weights.ndim != 2:
        # Fallback, though shouldn't happen
        weights = weights.reshape(5, -1)
        
    if weights.shape[0] != 5 and weights.shape[1] == 5:
        # If it's transposed (Items, Constraints), fix it
        weights = weights.T
        
    # Calculate the maximum weight for each item across all constraints
    # This represents the "bottleneck" capacity consumption for that item
    max_weights = weights.max(axis=0)
    
    # Calculate heuristic: Prize / MaxWeight
    # Add epsilon to avoid division by zero
    epsilon = 1e-9
    eta = total_prizes / (max_weights + epsilon)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    eta = np.maximum(eta, 1e-9)
    
    return eta
