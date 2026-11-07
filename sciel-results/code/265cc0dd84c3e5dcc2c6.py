import numpy as np

def heuristics(prizes, weights):
    # prizes and weights are (5, 100)
    # Calculate ratio for each constraint
    # Add epsilon to avoid division by zero
    eps = 1e-9
    ratios = prizes / (weights + eps)
    # The heuristic is the minimum ratio across constraints
    # This represents the efficiency with respect to the most constraining dimension
    heuristic = np.min(ratios, axis=0)
    return heuristic
