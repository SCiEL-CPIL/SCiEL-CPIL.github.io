"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Combined heuristic with aggressive long-edge suppression and topological guidance:
    1. Base: high-power inverse distance (5th power) to strongly penalize long/crossing edges.
    2. Topological boost: edges from a fast NN+2Opt reference tour are boosted.
    3. k-NN boost: edges to nearest neighbors are boosted.
    4. Planarity-aware penalty: additional penalty for edges that are geometrically
       likely to cross (long chords in 2D point sets).
    This guides ACO ants toward structurally valid, short, non-crossing tours.
    """
    start_time = time.time()
    deadline = start_time + 275.0  # Safety margin for 300s limit
    
    n = distance_matrix.shape[0]
    
    # 1. Generate a good reference tour using Nearest Neighbor
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited neighbor
        min_dist = np.inf
        next_node = -1
        for j in range(n):
            if not visited[j]:
                d = distance_matrix[current, j]
                if d < min_dist:
                    min_dist = d
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # 2. Improve with 2-Opt (limited iterations to save time)
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i + 1) % n]]
        return total
    
    current_len = tour_length(tour)
    improved = True
    max_iter = 500
    iter_count = 0
    
    while improved and iter_count < max_iter:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for 2-opt move
                # Remove edges (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # Add edges (tour[i-1], tour[j]) and (tour[i], tour[j+1])
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                
                old_len = distance_matrix[a, b] + distance_matrix[c, d]
                new_len = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_len < old_len - 1e-10:
                    # Reverse segment i..j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    current_len -= old_len - new_len
                    improved = True
                    if time.time() > deadline:
                        break
        if time.time() > deadline:
            break
    
    # 3. Build the heuristic matrix
    # Base heuristic: 5th power inverse distance to very aggressively suppress long edges
    safe_dist = np.where(distance_matrix == 0, 1e-9, distance_matrix)
    base_heuristic = 1.0 / (safe_dist ** 5)
    
    # Boost edges in the reference tour
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        base_heuristic[u, v] *= 20.0
        base_heuristic[v, u] *= 20.0  # Symmetric boost for ACO
    
    # Boost k-NN edges (k=3)
    k = 3
    for i in range(n):
        # Get k nearest neighbors
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for j in nn_indices:
            base_heuristic[i, j] *= 4.0
    
    # Planarity-aware penalty: penalize edges that are very long relative to local neighborhood
    # For each edge (i,j), if the distance is much larger than the median distance from i,
    # apply an additional penalty to discourage crossing edges
    if time.time() < deadline - 10:
        for i in range(n):
            dists_i = distance_matrix[i].copy()
            dists_i[i] = 0
            median_dist_i = np.median(dists_i)
            if median_dist_i > 1e-9:
                for j in range(n):
                    if i != j and distance_matrix[i, j] > 2.5 * median_dist_i:
                        base_heuristic[i, j] *= 0.5  # Penalize long edges
    
    # Ensure no zeros or negatives
    base_heuristic = np.maximum(base_heuristic, 1e-9)
    
    return base_heuristic
# EVOLVE-BLOCK-END
