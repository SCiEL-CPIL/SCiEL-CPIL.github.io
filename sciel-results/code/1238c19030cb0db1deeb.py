import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    
    # Avoid division by zero
    safe_weight = total_weight + 1e-9
    ratio = total_prize / safe_weight
    
    # Normalize ratio to [0, 1] range roughly
    max_ratio = np.max(ratio)
    if max_ratio > 0:
        normalized_ratio = ratio / max_ratio
    else:
        normalized_ratio = np.ones_like(ratio)
        
    # Greedy Selection to identify "good" items
    # Sort indices by ratio descending
    order = np.argsort(-ratio)
    
    remaining_cap = np.full(5, 25.0)
    selected = np.zeros(100, dtype=bool)
    
    for idx in order:
        w = weights[:, idx]
        if np.all(w <= remaining_cap):
            remaining_cap -= w
            selected[idx] = True
            
    # Boost selected items
    # Factor 1.0 to 2.0. Let's try 1.5.
    heuristic = normalized_ratio.copy()
    heuristic[selected] *= 1.5
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
