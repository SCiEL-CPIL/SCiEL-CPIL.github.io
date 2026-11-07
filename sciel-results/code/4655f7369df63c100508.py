import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    d = distance_matrix
    # Use exponential decay of distance as the heuristic
    # Higher prize and smaller distance -> higher attractiveness
    # exp(-d) is in (0, 1] for d >= 0
    heuristic = prizes[None, :] * np.exp(-d)
    
    # Ensure non-finite and non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    return heuristic
