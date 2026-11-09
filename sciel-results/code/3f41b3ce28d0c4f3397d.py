import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Ensure shape
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Base heuristic: inverse distance squared
    # Power 2 is a standard choice for TSP/CVRP heuristics in ACO,
    # providing a strong but not overly greedy preference for short edges.
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 2.0)
    
    # Handle non-positive or non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
