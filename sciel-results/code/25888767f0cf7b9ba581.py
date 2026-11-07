import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Calculate total prize per item
    total_prizes = np.sum(prizes, axis=0)
    
    # Capacity is 25 for all constraints
    capacity = 25.0
    
    # Calculate relative weight per constraint (fraction of capacity used)
    # weights is (5, 100), capacity is scalar
    relative_weights = weights / capacity
    
    # Identify the bottleneck constraint for each item
    # This is the constraint that limits this item the most
    # If an item uses 10% of constraint A and 50% of constraint B, B is the bottleneck.
    bottleneck_weights = np.max(relative_weights, axis=0)
    
    # Heuristic: Prize / Bottleneck Weight
    # This penalizes items that are heavy in their most restrictive constraint
    eps = 1e-9
    heur = total_prizes / (bottleneck_weights + eps)
    
    # Ensure values are positive
    heur = np.maximum(heur, 1e-9)
    
    return heur
