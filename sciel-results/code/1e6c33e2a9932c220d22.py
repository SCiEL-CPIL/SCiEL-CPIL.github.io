import numpy as np

def heuristics(prizes, weights):
    """
    Computes heuristic values for items in a Multiple-Constraint Knapsack Problem.
    
    Parameters:
    prizes: numpy array of shape (num_constraints, num_items)
    weights: numpy array of shape (num_constraints, num_items)
    
    Returns:
    numpy array of shape (num_items,) with heuristic scores.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues (though spec says (5, 100))
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # Calculate total prize for each item
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the sum of weights across all constraints for each item.
    # This captures the total resource consumption, which is more informative
    # than just the max (bottleneck) constraint, as seen in inspiration 1 & 2.
    w_sum = np.sum(weights, axis=0)
    
    # Calculate heuristic: Prize / Total_Weight
    # This favors items with high total prize relative to their total resource usage.
    heuristic = total_prizes / (w_sum + 1e-9)
    
    # Ensure positive and finite values
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
