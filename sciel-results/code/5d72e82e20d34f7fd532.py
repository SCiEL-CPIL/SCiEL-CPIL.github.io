import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands)
    
    # d_i is column vector, d_j is row vector
    d_i = d[:, None]
    d_j = d[None, :]
    
    # Calculate sums to check validity
    sums = d_i + d_j
    valid = sums <= capacity
    
    # Calculate remaining space for each i
    remaining = capacity - d_i
    
    # Initialize heuristic matrix
    heuristic = np.zeros((n, n), dtype=float)
    
    # Calculate ratio d_j / remaining for valid pairs
    # We use np.divide to handle the mask
    # where=valid ensures we only divide where valid is True
    # out=heuristic ensures the result is stored there
    # For invalid pairs, heuristic remains 0.0
    
    # Note: remaining is never 0 because d_i <= 99 and capacity = 150.
    np.divide(d_j, remaining, out=heuristic, where=valid)
    
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # Our valid entries are positive (d_j >= 20, remaining <= 130, so ratio > 0).
    # Our invalid entries are 0.0.
    # So we can just return the matrix, and the evaluator will handle the 1e-9 replacement?
    # "Heuristic entries that are not positive become 1e-9" is part of the ACO description, 
    # implying the evaluator or the ACO loop does this. 
    # However, to be safe and explicit, I can do it here.
    # But if I do it here, I must ensure I don't break the interface.
    # The interface just says "returns an item-by-item matrix".
    # The text says "Heuristic entries that are not positive become 1e-9".
    # This usually happens inside the ACO probability calculation.
    # I will return the raw values. If I want to be safe, I can set invalid to 1e-9 directly.
    
    heuristic[~valid] = 1e-9
    
    return heuristic
