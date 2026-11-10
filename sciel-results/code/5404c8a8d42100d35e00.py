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
        # Find nearest unvisited neighbor
        unvisited = ~visited
        unvisited[current] = True  # Exclude current
        nearest = np.argmin(np.where(unvisited, dist[current], np.inf))
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
    return tour


def _two_opt_tour(tour, dist, max_passes=20):
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
                # Calculate delta for reversing tour[i+1..j]
                a, b = best_tour[i], best_tour[i + 1]
                c, d = best_tour[j], best_tour[(j + 1) % n]
                old_len = dist[a][b] + dist[c][d]
                new_len = dist[a][c] + dist[b][d]
                if new_len < old_len - 1e-12:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    improved = True
    return best_tour


def _knn_graph(dist, k=6):
    """Build k-nearest-neighbor graph. Returns boolean adjacency matrix."""
    n = dist.shape[0]
    knn = np.zeros((n, n), dtype=bool)
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        idx = np.argsort(dist[i])
        for j in idx[1:k+1]:
            knn[i, j] = True
            knn[j, i] = True
    return knn


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP.
    
    Strategy:
    1. Build a high-quality reference tour using NN + 2-opt
    2. Compute k-NN graph for local structure
    3. Base: inverse distance to power 2.0
    4. Boost: edges in reference tour get high weight
    5. Penalty: edges NOT in k-NN graph and NOT in reference tour get reduced
    
    This combines global tour structure with local proximity constraints.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Step 1: Build reference tour
    nn_tour = _nearest_neighbor_tour(distance_matrix)
    ref_tour = _two_opt_tour(nn_tour, distance_matrix, max_passes=20)
    
    # Step 2: Extract edges from reference tour
    ref_edge_set = set()
    for idx in range(n):
        u = ref_tour[idx]
        v = ref_tour[(idx + 1) % n]
        ref_edge_set.add((min(u, v), max(u, v)))
    
    # Step 3: Build k-NN graph
    knn_adj = _knn_graph(distance_matrix, k=6)
    
    # Step 4: Base heuristic - inverse distance to power 2.0
    base = 1.0 / (distance_matrix + 1e-9) ** 2.0
    
    # Step 5: Create boost/penalty matrix
    # - Reference tour edges: high boost (10.0)
    # - k-NN edges (not in ref tour): moderate boost (2.0)
    # - Other edges: penalty (0.3)
    boost = np.ones((n, n), dtype=np.float64)
    
    # Mark reference tour edges
    ref_edge_boost = np.zeros((n, n), dtype=np.float64)
    for (u, v) in ref_edge_set:
        ref_edge_boost[u, v] = 1.0
        ref_edge_boost[v, u] = 1.0
    
    # Combine: ref edges get 10, k-NN edges get 2, others get 0.3
    boost = np.where(ref_edge_boost > 0, 10.0,
           np.where(knn_adj, 2.0, 0.3))
    
    # Ensure diagonal is 0 (self-loops)
    np.fill_diagonal(boost, 0.0)
    
    # Combine all components
    heuristic = base * boost
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    np.fill_diagonal(heuristic, 0.0)  # Self-loops should be 0
    
    return heuristic
# EVOLVE-BLOCK-END
