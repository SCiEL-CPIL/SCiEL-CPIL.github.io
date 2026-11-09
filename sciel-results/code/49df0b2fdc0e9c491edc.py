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
    
    # Calculate the maximum weight across constraints for each item.
    # This captures the "bottleneck" constraint for each item.
    # Items with a low max weight are less likely to be blocked by a single tight constraint.
    max_weights = np.max(weights, axis=0)
    
    # Calculate heuristic: Prize / Max_Weight
    # This favors items with high total prize that do not heavily stress any single constraint.
    heuristic = total_prizes / (max_weights + 1e-9)
    
    # Ensure positive and finite values
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
