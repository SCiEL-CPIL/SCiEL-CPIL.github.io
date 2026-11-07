"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_tour(dist):
    n = len(dist)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best = -1
        best_d = float('inf')
        for j in range(n):
            if not visited[j] and dist[last, j] < best_d:
                best_d = dist[last, j]
                best = j
        tour.append(best)
        visited[best] = True
    return tour

def _two_opt(tour, dist):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour

def heuristics(distance_matrix):
    start = time.time()
    deadline = start + 280  # safety margin
    
    dist = np.asarray(distance_matrix, dtype=np.float64)
    n = dist.shape[0]
    
    # Ensure symmetric and handle diagonal
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    
    # Compute greedy tour and 2-opt improve
    tour = _greedy_tour(dist)
    if time.time() < deadline - 10:
        tour = _two_opt(tour, dist)
    
    # Build set of edges in the improved tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
    
    # Base heuristic: inverse distance with power
    # Use a small epsilon to avoid division by zero
    safe_dist = dist + 1e-9
    # Power of 2 emphasizes short edges significantly
    base_heuristic = 1.0 / (safe_dist ** 2)
    
    # Boost edges in the 2-opt improved tour
    # This guides the ACO toward a known-good structure
    boost_factor = 5.0
    heuristic = base_heuristic.copy()
    heuristic[tour_edges] *= boost_factor
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
