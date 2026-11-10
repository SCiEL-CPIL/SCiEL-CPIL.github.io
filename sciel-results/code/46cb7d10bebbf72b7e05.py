"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP.
    
    Uses a multi-component heuristic:
    1. Inverse distance (base attractiveness)
    2. Nearest-neighbor graph boost (edges in k-NN graph are more attractive)
    3. Long-edge penalty (edges much longer than local scale are penalized)
    """
    n = distance_matrix.shape[0]
    deadline = time.time() + 280  # Safety margin before 330s limit
    
    # Precompute local scale for each node: median distance to 10 nearest neighbors
    local_scales = np.zeros(n)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        k_local = min(10, n - 1)
        sorted_dists = np.sort(dists)[:k_local]
        local_scales[i] = np.median(sorted_dists) if len(sorted_dists) > 0 else 1.0
    
    # Base: inverse distance squared (moderate suppression of long edges)
    base = 1.0 / (distance_matrix + 1e-9) ** 2
    
    # Component 1: k-NN graph boost
    # Edges that appear in the k-nearest-neighbor graph are more attractive
    k = 4
    knn_boost = np.zeros_like(distance_matrix)
    for i in range(n):
        # Get k nearest neighbors of i (excluding self)
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for j in nn_indices:
            knn_boost[i, j] = 1.0
            knn_boost[j, i] = 1.0
    
    # Scale knn_boost to be a modest multiplier
    knn_multiplier = 1.0 + 0.3 * knn_boost
    
    # Component 2: Long-edge penalty using local scale
    # Penalize edges that are much longer than the local scale of their endpoints
    long_edge_penalty = np.ones_like(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            dist_ij = distance_matrix[i, j]
            # Use the average local scale of both endpoints
            avg_local_scale = 0.5 * (local_scales[i] + local_scales[j])
            ratio = dist_ij / (avg_local_scale + 1e-9)
            if ratio > 1.5:
                penalty = 1.0 / (1.0 + 0.4 * (ratio - 1.5))
                long_edge_penalty[i, j] = penalty
                long_edge_penalty[j, i] = penalty
    
    # Combine all components
    heuristic = base * knn_multiplier * long_edge_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
