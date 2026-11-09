"""Rewritten program: node-centric greedy ratio heuristic for ACO orienteering."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix of the same shape as distance_matrix.
    
    The heuristic models the attractiveness of moving to node j from node i
    as proportional to prizes[j] / (dist[i,j] + dist[j,0]).
    
    This treats the edge choice as maximizing immediate prize density relative
    to the total length cost of visiting j and then returning to the depot (node 0).
    Nodes with high prizes that are close to both the current node and the depot
    are more attractive.
    
    Non-positive or non-finite entries are clipped to 1e-9.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Build the heuristic: attractiveness[i, j] = prizes[j] / (dist[i,j] + dist[j,0])
    # We want a matrix of shape (n, n)
    
    # dist[j, 0] for each j: shape (n,)
    dist_to_depot = distance_matrix[:, 0]  # shape (n,)
    
    # For each pair (i, j): numerator = prizes[j], denominator = dist[i,j] + dist[j,0]
    # numerator matrix: shape (n, n) where each row is prizes (broadcast)
    numerator = prizes[None, :]  # shape (1, n) -> broadcast to (n, n)
    
    # denominator: distance_matrix + dist_to_depot[None, :]
    # distance_matrix[i, j] + dist_to_depot[j]
    denominator = distance_matrix + dist_to_depot[None, :]  # shape (n, n)
    
    # Avoid division by zero or negative values
    denominator = np.where(denominator > 1e-12, denominator, 1e-12)
    
    heuristic = numerator / denominator
    
    # Ensure non-negative: prizes are in [0,1], so numerator >= 0
    # Clip to ensure positive values; non-positive becomes 1e-9
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Make sure it's a proper array of the same shape as distance_matrix
    heuristic = np.asarray(heuristic, dtype=np.float64)
    
    if heuristic.shape != distance_matrix.shape:
        raise ValueError("Heuristic shape mismatch")
    
    return heuristic
# EVOLVE-BLOCK-END
