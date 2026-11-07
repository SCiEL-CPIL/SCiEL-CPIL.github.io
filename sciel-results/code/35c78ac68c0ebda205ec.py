import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix for ant colony optimization of CVRP.
    Uses an inverse distance power law with exponent 1.5 to balance 
    greediness (short edges) and exploration (pheromone influence).
    """
    # Ensure we work with floats
    d = distance_matrix.astype(float)
    
    # Heuristic: Inverse distance to the power of 1.5
    # This is less aggressive than d^-2 (current best) but more than d^-1 (baseline).
    # It aims to prevent ultra-short edges from completely dominating the pheromone trail.
    heuristic = 1.0 / (d**1.5 + 1e-9)
    
    # Set diagonal to 0 to prevent self-loops
    np.fill_diagonal(heuristic, 0.0)
    
    # Replace non-positive or non-finite values with 1e-9
    # This ensures the matrix is valid for probability calculations
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
