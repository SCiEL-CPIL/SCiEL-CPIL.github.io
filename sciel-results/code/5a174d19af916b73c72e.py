import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard
    t0 = time.time()
    deadline = 300.0
    
    # Convert to numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure shapes are correct (5, 100)
    # The problem states shape (5, 100)
    
    # Calculate total prize for each item
    # Sum across the 5 constraints
    total_prizes = prizes.sum(axis=0)
    
    # Calculate normalized total weight for each item
    # Capacity is 25 for each constraint
    capacity = 25.0
    normalized_weights = weights / capacity
    
    # Sum normalized weights across constraints
    total_normalized_weights = normalized_weights.sum(axis=0)
    
    # Avoid division by zero
    eps = 1e-9
    total_normalized_weights = np.maximum(total_normalized_weights, eps)
    
    # Heuristic: Total Prize / Total Normalized Weight
    # This represents the value density of the item relative to the total resource capacity
    heuristic = total_prizes / total_normalized_weights
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check deadline (though this calculation is very fast)
    if time.time() - t0 > deadline - 5.0:
         # Fallback if somehow slow (extremely unlikely)
         pass
         
    return heuristic
