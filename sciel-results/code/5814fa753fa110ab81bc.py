"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    This heuristic combines:
    1. Inverse distance (standard local bias)
    2. Clarke-Wright Savings factor (global structural bias)
    
    The savings term rewards edges that connect two nodes which are close to 
    each other relative to their distances from the depot. This encourages 
    the formation of routes that minimize total travel distance by reducing 
    the "penalty" of returning to the depot for each pair of nodes.
    
    Returns a matrix where larger values are more attractive.
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
        
    # Ensure inputs are numpy arrays for efficient computation
    D = np.asarray(distance_matrix, dtype=float)
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    
    # Base heuristic: Inverse distance
    # Larger values for shorter distances
    inv_dist = 1.0 / (D + eps)
    
    # Savings Heuristic:
    # Savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    # This represents the distance saved by going i -> j directly instead of i -> 0 -> j.
    
    # Distances from depot (node 0) to all other nodes
    d_depot = D[0, :]
    
    # Create savings matrix
    # d_depot[i] is D[0, i]
    # d_depot[j] is D[0, j]
    # D[i, j] is the direct distance
    
    # Broadcasting:
    # d_depot_i shape: (n, 1)
    # d_depot_j shape: (1, n)
    d_i = d_depot[:, np.newaxis]
    d_j = d_depot[np.newaxis, :]
    
    savings = d_i + d_j - D
    
    # Only positive savings are beneficial. Negative savings mean the direct
    # path is longer than going via depot (geometrically impossible in Euclidean 
    # space for triangle inequality, but possible if matrix is not strictly metric 
    # or due to floating point errors, though Euclidean is metric).
    # In Euclidean space, triangle inequality holds: D[i,j] <= D[0,i] + D[0,j].
    # So savings >= 0.
    
    # Normalize savings to be in range [0, 1] roughly
    # Max possible savings for a pair is d_i + d_j (if D[i,j] -> 0)
    # Min is 0 (if D[i,j] = d_i + d_j, i.e., collinear with depot)
    
    denom = d_i + d_j + eps
    norm_savings = np.maximum(0.0, savings) / denom
    
    # Combine: 
    # We want to boost the inverse distance by the savings factor.
    # If savings is high (nodes are close to each other relative to depot), 
    # the edge is very attractive.
    # Factor ranges from 1.0 (no savings) to ~2.0 (max savings).
    heuristic = inv_dist * (1.0 + norm_savings)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
