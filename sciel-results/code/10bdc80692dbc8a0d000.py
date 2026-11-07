import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Ensure positive values to avoid division by zero
    eps = 1e-9
    p = prizes + eps
    w = weights + eps
    
    # New Approach: Global Density
    # Sum prizes across constraints to get total value per item
    # Sum weights across constraints to get total resource consumption per item
    total_prizes = np.sum(p, axis=0)
    total_weights = np.sum(w, axis=0)
    
    # Heuristic is total prize divided by total weight
    heur = total_prizes / total_weights
    
    # Ensure positive and finite
    heur = np.maximum(heur, 1e-9)
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    return heur
