import numpy as np

def heuristics(prizes, weights):
    # prizes and weights are (5, 100), uniform in [0, 1]
    # Normalize weights per constraint to [0, 1]
    w_min = np.min(weights, axis=1, keepdims=True)
    w_max = np.max(weights, axis=1, keepdims=True)
    w_range = w_max - w_min
    # Avoid division by zero
    w_range = np.where(w_range < 1e-12, 1.0, w_range)
    weights_norm = (weights - w_min) / w_range  # shape (5, 100)
    
    # For each item, find the maximum normalized weight across constraints
    # This represents the "worst-case" resource consumption relative to range
    max_norm_weight = np.max(weights_norm, axis=0)  # shape (100,)
    
    # Normalize prizes per constraint to [0, 1] as well
    p_min = np.min(prizes, axis=1, keepdims=True)
    p_max = np.max(prizes, axis=1, keepdims=True)
    p_range = p_max - p_min
    p_range = np.where(p_range < 1e-12, 1.0, p_range)
    prizes_norm = (prizes - p_min) / p_range  # shape (5, 100)
    
    # Sum normalized prizes across constraints to get total value
    total_prize_norm = np.sum(prizes_norm, axis=0)  # shape (100,)
    
    # Heuristic: product of normalized prize and inverse of max normalized weight
    # Add epsilon to avoid division by zero
    eps = 1e-9
    heuristic = total_prize_norm / (max_norm_weight + eps)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
