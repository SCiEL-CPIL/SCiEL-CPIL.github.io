"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering.
    
    The heuristic should guide ants toward nodes that:
    1. Have high prizes
    2. Are close to the current node (small d[i,j])
    3. Are close to the origin (small d[j,0]) so they fit within max_len
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # d_j0: distance from each node j to origin (shape: (1, n) for broadcasting)
    d_j0 = distance_matrix[:, 0].reshape(1, -1)
    
    # Prize component: use log(1+prize) to compress the range and avoid dominance by high prizes
    prize_component = np.log1p(prizes)[None, :]  # shape: (1, n)
    
    # Distance penalty: penalize both d[i,j] and d[j,0]
    # The constraint is length + d[i,j] + d[j,0] <= max_len
    # So both distances matter. Use a combined penalty.
    # d_ij has shape (n, n), d_j0 has shape (1, n)
    d_ij = distance_matrix  # shape: (n, n)
    
    # Combined distance: we want to penalize nodes that are far from both i and origin
    # Use a weighted combination: alpha * d[i,j] + beta * d[j,0]
    # With max_len=3, both distances are important
    combined_dist = 0.5 * d_ij + 0.5 * d_j0  # shape: (n, n)
    
    # Heuristic: prize / (combined_dist + eps)
    # This rewards high prizes and small distances
    heuristic = prize_component / (combined_dist + eps)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
