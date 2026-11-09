"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Node-centric greedy ratio: attractiveness of moving to node j from i
    # is proportional to prizes[j] / (dist[i,j] + dist[j,0])
    # This treats the edge choice as maximizing immediate prize density
    # relative to the remaining length budget.
    
    # dist[j, 0] is the cost to return to depot from node j
    return_dist = distance_matrix[:, 0]  # shape (n,)
    
    # For each pair (i, j), heuristic[i, j] = prizes[j] / (dist[i,j] + dist[j,0])
    # We want to emphasize high prizes and low total path cost
    # Add small epsilon to avoid division by zero
    denom = distance_matrix + return_dist[None, :] + 1e-9
    numer = prizes[None, :] + 1e-9
    
    # The heuristic matrix
    H = numer / denom
    
    # Ensure positive values
    H = np.maximum(H, 1e-9)
    
    # Make symmetric? No, the heuristic can be asymmetric since dist[i,j] != dist[j,i]
    # But distance_matrix is symmetric in Euclidean, so denom is not symmetric
    # because return_dist[None, :] adds to each column j the same value for all i
    # Actually denom[i,j] = dist[i,j] + dist[j,0], which is not symmetric
    # This is fine - the heuristic can be asymmetric
    
    return H
# EVOLVE-BLOCK-END
