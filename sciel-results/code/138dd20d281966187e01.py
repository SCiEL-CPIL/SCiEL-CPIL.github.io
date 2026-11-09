import numpy as np
import time

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
        
    num_constraints, num_items = weights.shape
    
    # Calculate total prize for each item
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the sum of weights for each constraint across all items
    # S_c = sum_{j=1}^{N} W_{jc}
    constraint_loads = np.sum(weights, axis=1)
    
    # Determine the "tightness" or influence of each constraint.
    # Constraints with higher total load are tighter (items are heavier on average).
    # We use the load itself as the weight for that constraint in the effective cost.
    # This makes the heuristic instance-aware.
    
    # Avoid division by zero if a constraint has 0 total weight (unlikely with uniform [0,1] but safe)
    total_load = np.sum(constraint_loads)
    if total_load < 1e-9:
        # Fallback to uniform weights if all constraints are empty
        constraint_weights = np.ones(num_constraints) / num_constraints
    else:
        constraint_weights = constraint_loads / total_load
        
    # Calculate the effective weight for each item
    # effective_weight_i = sum_c (constraint_weight_c * W_{ic})
    # This is a weighted sum where constraints that are globally tighter contribute more to the cost.
    effective_weights = np.sum(constraint_weights[:, np.newaxis] * weights, axis=0)
    
    # Calculate heuristic: Total Prize / Effective Weight
    # Add epsilon to denominator to avoid division by zero
    heuristic = total_prizes / (effective_weights + 1e-9)
    
    # Ensure positive and finite values
    heuristic = np.maximum(heuristics, 1e-9)
    heuristic = np.where(np.isfinite(heuristics), heuristic, 1e-9)
    
    return heuristic
