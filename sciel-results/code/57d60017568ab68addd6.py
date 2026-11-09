import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Calculate sum of pairs
    pair_sums = demands[:, None] + demands[None, :]
    
    # Calculate remaining capacity (waste)
    waste = capacity - pair_sums
    
    # Valid pairs are those where waste >= 0
    valid = waste >= 0
    
    # Heuristic: Inverse of waste (shifted by 1 to avoid division by zero and soften peak)
    # Scaled by capacity to keep magnitudes reasonable
    # If waste is 0, H = capacity/1 = capacity
    # If waste is large, H is small
    H = np.where(valid, capacity / (waste + 1.0), 0.0)
    
    # Ensure no negative or zero values for valid pairs if we want to be strict, 
    # but the task says non-positive becomes 1e-9 in the ACO loop.
    # Returning 0.0 for invalid pairs is standard.
    
    return H
