import numpy as np

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    # Calculate total prize per item
    # The problem states prizes shape (5, 100). 
    # "highest total prize" implies the objective is the sum of prizes.
    total_prize = np.sum(prizes, axis=0)
    
    # Normalize prize to [0, 1] to avoid scale issues
    max_prize = np.max(total_prize)
    if max_prize > 0:
        norm_prize = total_prize / max_prize
    else:
        norm_prize = np.ones_like(total_prize)
        
    # Calculate the bottleneck weight per item
    # The constraint that is most tightly bound by item i is the one with max weight
    # weights shape is (5, 100), values in [0, 1]
    # Capacity is 25, but since we are looking at relative cost/bottleneck, 
    # and all constraints have same capacity, the relative "expensiveness" 
    # in terms of limiting future choices is determined by the max weight.
    # We use the raw max weight from [0,1] as the "normalized weight" 
    # since the input is already normalized to [0,1].
    
    max_weight = np.max(weights, axis=0)
    
    # Heuristic: normalized prize * inverse of max weight
    # Items with high prize and low max weight are most attractive
    eps = 1e-9
    heuristic = norm_prize / (max_weight + eps)
    
    # Ensure finite and positive
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
