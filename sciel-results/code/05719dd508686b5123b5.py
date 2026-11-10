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
        local_scales[i] = np.median(sorted_dists)
    
    # Base: inverse distance squared (moderate suppression of long edges)
    base = 1.0 / (distance_matrix + 1e-9) ** 2
    
    # Component 1: Smooth k-NN graph boost
    # Use a soft weighting based on rank among neighbors
    k = 6
    knn_boost = np.zeros_like(distance_matrix)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        # Rank-based weight: nearest gets 1.0, k-th gets 0.0
        nn_indices = np.argsort(dists)[:k]
        for rank, j in enumerate(nn_indices):
            weight = 1.0 - (rank / k)
            knn_boost[i, j] = weight
            knn_boost[j, i] = max(knn_boost[j, i], weight)
    
    # Scale knn_boost to be a modest multiplier
    knn_multiplier = 1.0 + 0.4 * knn_boost
    
    # Component 2: Long-edge penalty based on local scale
    # Precompute using vectorized operations
    crossing_penalty = np.ones_like(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = distance_matrix[i, j]
            # Use the smaller local scale for a conservative penalty
            local_scale = min(local_scales[i], local_scales[j])
            if local_scale < 1e-9:
                local_scale = 1e-9
            ratio = dist_ij / local_scale
            if ratio > 1.5:
                # Smooth penalty: exponential decay for long edges
                penalty = np.exp(-0.3 * (ratio - 1.5))
                crossing_penalty[i, j] = penalty
                crossing_penalty[j, i] = penalty
    
    # Combine all components
    heuristic = base * knn_multiplier * crossing_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
