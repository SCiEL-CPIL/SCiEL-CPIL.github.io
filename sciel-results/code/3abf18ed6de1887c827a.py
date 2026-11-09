import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    d = np.array(distance_matrix, dtype=float)
    demands = np.array(demands, dtype=float)
    n = d.shape[0]
    
    # Base distance heuristic with tuned power
    # Power 3.5 provides a good balance between exploration and exploitation
    # compared to power 4 which was too aggressive
    heuristic = np.zeros_like(d)
    mask = d > 0
    heuristic[mask] = 1.0 / (d[mask] ** 3.5)
    
    # Set diagonal to 0 to discourage self-loops
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure any non-positive values become 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
