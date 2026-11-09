"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a static heuristic matrix for ACO orienteering.
    
    The heuristic favors nodes with high prize relative to the cost of
    visiting them and returning to the start. We use:
        h[i, j] = prize[j] / (d[i,j] + d[j,0] + eps)
    
    This captures the idea that a node is attractive if it offers high prize
    for low additional path length (including return trip).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # Cost of going from i to j and then back to 0
    # d[i,j] + d[j,0]
    return_cost = distance_matrix[:, 0][None, :]  # d[j, 0] for each j, shape (1, n)
    # d[i,j] has shape (n, n)
    total_cost = distance_matrix + return_cost  # shape (n, n): d[i,j] + d[j,0]
    
    # Heuristic: prize[j] / total_cost[i,j]
    # prize[j] broadcast along axis 0
    h = prizes[None, :] / (total_cost + eps)
    
    # Ensure positive values; replace non-positive with small value
    h = np.where(h > 0, h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
