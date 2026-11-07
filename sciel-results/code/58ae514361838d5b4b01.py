"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Prize term: strong gradient toward high-value nodes
    # Use max with small floor to avoid zero for node 0 (prize=0)
    prize_term = np.maximum(prizes, 1e-6) ** 3.0  # alpha=3
    
    # Distance cost: incorporate both d[i,j] and d[j,0] (round-trip feasibility)
    # d[i,j] is distance_matrix[i,j], d[j,0] is distance_matrix[j,0]
    # Shape: distance_matrix is (n, n), so d[i,j] is DM[i,j]
    # d[j,0] for all j is DM[:, 0], reshaped to (1, n) to broadcast over i
    d_ij = distance_matrix  # (n, n)
    d_j0 = distance_matrix[:, 0][None, :]  # (1, n), broadcast over i
    
    # Round-trip cost with power penalty
    # Use (1 + d) to avoid zero, then power beta
    beta = 1.5
    dist_cost = (1.0 + d_ij) ** beta + (1.0 + d_j0) ** beta  # (n, n)
    
    # Heuristic: prize / distance_cost
    heuristic = prize_term[None, :] / dist_cost  # (n, n)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
