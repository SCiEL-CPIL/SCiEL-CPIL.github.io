"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Heuristic based on marginal gain: prize weighted by remaining budget slack
    # For edge (i, j), the slack after visiting j is max_len - d[i,j] - d[j,0]
    # We weight prize[j] by this slack to prioritize nodes that fit well in the tour
    
    # Compute slack for each edge (i, j): remaining budget after adding j
    # d[i,j] is the distance from i to j, d[j,0] is distance from j back to 0
    # Slack = max_len - d[i,j] - d[j,0]
    # Only positive slack means the edge is feasible
    
    slack = max_len - distance_matrix - distance_matrix[:, 0][np.newaxis, :]
    
    # Clip slack to be non-negative (infeasible edges get zero slack)
    slack = np.maximum(slack, 0.0)
    
    # Prize density: prize[j] / (distance cost) but weighted by slack
    # We want high prize, good slack, and reasonable distance
    
    # Base heuristic: prize[j] * slack[i,j] / (distance[i,j] + eps)^beta
    eps = 1e-3
    alpha = 2.0
    beta = 1.0
    
    prize_factor = np.power(prizes + 1e-9, alpha)  # shape (n,)
    dist_factor = np.power(distance_matrix + eps, beta)  # shape (n, n)
    
    # Heuristic: for each (i, j), weight by prize[j] * slack[i,j] / dist[i,j]^beta
    # This makes edges to high-prize nodes with good remaining budget more attractive
    heuristic = prize_factor[np.newaxis, :] * slack / dist_factor
    
    # Add a small base term to avoid zero entries and ensure all entries are positive
    # This helps with exploration in early iterations
    base_heuristic = prize_factor[np.newaxis, :] / (distance_matrix + eps)
    heuristic = 0.1 * base_heuristic + 0.9 * heuristic
    
    # Ensure positive finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
