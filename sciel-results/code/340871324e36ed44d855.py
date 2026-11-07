import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    eps = 1e-9
    
    # Constraint scarcity: total weight per constraint (higher = more binding)
    # Normalize so that average scarcity is 1
    constraint_total = weights.sum(axis=1)  # (5,)
    avg_total = constraint_total.mean()
    scarcity = constraint_total / (avg_total + eps)  # (5,)
    
    # Per-constraint, per-item: prize / (weight * scarcity)
    # This penalizes items in scarce constraints more
    scaled_weights = weights * scarcity[:, np.newaxis]  # (5, 100)
    heuristic = prizes / (scaled_weights + eps)
    
    return heuristic
