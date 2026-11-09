import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for ACO.
    
    Args:
        prizes: array of shape (n,)
        distance_matrix: array of shape (n, n)
        max_len: float
    
    Returns:
        heuristic: array of shape (n, n)
    """
    n = len(prizes)
    
    # Ensure inputs are floats
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Calculate the "detour" cost for each edge (i, j)
    # Cost of going i -> j -> 0 compared to i -> 0
    # detour[i, j] = d[i, j] + d[j, 0] - d[i, 0]
    # This represents the additional distance incurred by visiting j before returning home.
    
    # d_j_0 is the distance from each node to the depot (node 0)
    d_j_0 = distance_matrix[:, 0]
    # d_i_0 is the distance from each node to the depot (node 0)
    d_i_0 = distance_matrix[:, 0]
    
    # detour[i, j] = dist[i, j] + d_j_0[j] - d_i_0[i]
    # We want to compute this for all i, j.
    # dist[i, j] is distance_matrix[i, j]
    # d_j_0[j] needs to be broadcast along rows (axis 0) -> shape (1, n)
    # d_i_0[i] needs to be broadcast along cols (axis 1) -> shape (n, 1)
    
    d_j_0_row = d_j_0[None, :] # Shape (1, n)
    d_i_0_col = d_i_0[:, None] # Shape (n, 1)
    
    detour = distance_matrix + d_j_0_row - d_i_0_col
    
    # Due to floating point or triangle inequality nuances, detour should be >= 0.
    # Clamp to a small positive value to avoid division by zero or negative denominators.
    eps = 1e-9
    detour_safe = np.maximum(detour, eps)
    
    # Heuristic: Prize at destination / Detour Cost
    # We use a power of 1 (alpha=1) first.
    # If we want to be more aggressive on distance, we can square the denominator.
    # Let's try alpha=1 first as it's the direct generalization of the baseline which used dist (which is roughly related to detour).
    
    heuristic = prizes[None, :] / detour_safe
    
    # Ensure no zeros (task says non-positive become 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
