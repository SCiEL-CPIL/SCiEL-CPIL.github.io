"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Parameters
    alpha = 2.5
    beta = 1.2
    gamma = 2.0
    eps = 1e-9
    
    # Extract distance to origin for each node
    dist_to_origin = distance_matrix[:, 0]
    
    # Calculate heuristic components
    # prize_term: shape (N,) -> broadcast to (N, N) along axis 0
    prize_term = (prizes + eps) ** alpha
    
    # dist_ij_term: shape (N, N)
    dist_ij_term = (distance_matrix + eps) ** beta
    
    # dist_j0_term: shape (N,) -> broadcast to (N, N) along axis 1
    dist_j0_term = (dist_to_origin + eps) ** beta
    
    # Slack term: remaining budget after visiting node j from i
    # slack_ij = max_len - d[i,j] - d[j,0]
    # Only positive slack is useful
    slack = max_len - distance_matrix - dist_to_origin[None, :]
    slack = np.maximum(slack, eps)
    slack_term = slack ** gamma
    
    # Neighborhood density: for each node j, how many other high-prize nodes are nearby
    # This rewards edges leading to clusters of high-value nodes
    # Compute for each pair (i,j): sum of prizes of nodes k that are close to j
    # Simplified: use prize of j weighted by local density
    # density_j = sum over k of prize[k] * exp(-d[j,k] / scale)
    scale = 0.3
    density = np.zeros(n)
    for j in range(n):
        # Distance from j to all other nodes
        dists_j = distance_matrix[j, :]
        weights = np.exp(-dists_j / scale)
        density[j] = np.sum(prizes * weights)
    
    # Broadcast density to (N, N) along axis 0 (depends only on j)
    density_term = (density + eps) ** 0.5
    
    # Combine all terms
    heuristic = (prize_term[None, :] * density_term[None, :] * slack_term) / (dist_ij_term * dist_j0_term[None, :])
    
    # Ensure positive finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
