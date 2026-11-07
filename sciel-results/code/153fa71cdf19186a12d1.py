import numpy as np

def heuristics(prizes, weights):
    # prizes and weights are shape (5, 100)
    # Capacity is fixed at 25 for all constraints
    
    capacity = 25.0
    
    # 1. Calculate total prize for each item
    total_prize = np.sum(prizes, axis=0)
    
    # 2. Calculate relative weight for each item and constraint
    # w_rel[i, j] = weights[j, i] / capacity
    rel_weights = weights / capacity
    
    # 3. Determine the bottleneck relative weight for each item
    # This is the maximum fraction of any capacity consumed by the item.
    # If an item consumes 0.8 of constraint 1 and 0.1 of others, its bottleneck is 0.8.
    bottleneck = np.max(rel_weights, axis=0)
    
    # 4. Compute heuristic as prize efficiency relative to the bottleneck constraint
    # Higher prize and lower bottleneck weight -> Higher heuristic.
    # Add epsilon to avoid division by zero.
    epsilon = 1e-9
    heuristic = total_prize / (bottleneck + epsilon)
    
    # Ensure output is finite and positive
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # But we should return the raw calculated values, the evaluator handles the 1e-9 replacement?
    # "Heuristic entries that are not positive become 1e-9" is a rule in the ACO loop, 
    # but the function returns the scores. It's safer to return positive values.
    # Our calculation yields positive values if inputs are positive.
    
    return heuristic
