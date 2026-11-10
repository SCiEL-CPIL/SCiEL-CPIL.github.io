"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dist):
    """Construct a nearest-neighbor tour starting from node 0."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        unvisited = ~visited
        unvisited[current] = True
        nearest = np.argmin(np.where(unvisited, dist[current], np.inf))
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    return tour


def _two_opt_tour(tour, dist, max_passes=5):
    """Improve tour using 2-opt local search."""
    n = len(tour)
    if n < 4:
        return tour
    
    best_tour = list(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                old_len = dist[a][b] + dist[c][d]
                new_len = dist[a][c] + dist[b][d]
                if new_len < old_len - 1e-12:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    improved = True
    return best_tour


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP.
    
    Combines:
    1. Inverse distance (moderate power) as base
    2. Reference tour boost (NN + 2-opt) - strong attraction for good edges
    3. k-NN graph boost
    4. Local scale penalty for long edges
    """
    n = distance_matrix.shape[0]
    deadline = time.time() + 280
    
    # Build reference tour: NN + 2-opt
    nn_tour = _nearest_neighbor_tour(distance_matrix)
    ref_tour = _two_opt_tour(nn_tour, distance_matrix, max_passes=5)
    
    # Create reference tour edge mask
    ref_tour_edges = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = ref_tour[i]
        b = ref_tour[(i + 1) % n]
        ref_tour_edges[a, b] = 1.0
        ref_tour_edges[b, a] = 1.0
    
    # Base: inverse distance with power 3 (aggressive suppression of long edges)
    base = 1.0 / (distance_matrix + 1e-9) ** 3
    
    # Component 1: Strong reference tour boost
    # Edges in the reference tour get a large multiplier
    ref_multiplier = 1.0 + 2.0 * ref_tour_edges
    
    # Component 2: k-NN graph boost (moderate)
    k = 5
    knn_boost = np.zeros_like(distance_matrix)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for rank, j in enumerate(nn_indices):
            weight = 1.0 - (rank / k)
            knn_boost[i, j] = weight
            knn_boost[j, i] = max(knn_boost[j, i], weight)
    
    knn_multiplier = 1.0 + 0.3 * knn_boost
    
    # Component 3: Local scale penalty
    # Precompute local scales
    local_scales = np.zeros(n)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        k_local = min(10, n - 1)
        sorted_dists = np.sort(dists)[:k_local]
        local_scales[i] = np.median(sorted_dists)
    
    # Long-edge penalty
    crossing_penalty = np.ones_like(distance_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = distance_matrix[i, j]
            local_scale = min(local_scales[i], local_scales[j])
            if local_scale < 1e-9:
                local_scale = 1e-9
            ratio = dist_ij / local_scale
            if ratio > 1.5:
                penalty = np.exp(-0.4 * (ratio - 1.5))
                crossing_penalty[i, j] = penalty
                crossing_penalty[j, i] = penalty
    
    # Combine all components
    heuristic = base * ref_multiplier * knn_multiplier * crossing_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
