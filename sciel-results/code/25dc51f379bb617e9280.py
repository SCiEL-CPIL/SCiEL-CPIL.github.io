"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nearest_neighbor_tour(start, dist):
    n = dist.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        current = tour[-1]
        d_masked = dist[current].copy()
        d_masked[visited] = float('inf')
        next_node = np.argmin(d_masked)
        tour.append(next_node)
        visited[next_node] = True
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
                u, v = tour[i], tour[i+1]
                x, y = tour[j], tour[j+1]
                
                d_old = dist[u, v] + dist[x, y]
                d_new = dist[u, x] + dist[v, y]
                
                if d_new < d_old - 1e-9:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
    return tour

def _tour_length(tour, dist):
    n = len(tour)
    total = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total += dist[u, v]
    return total

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    best_tour = None
    best_len = float('inf')
    
    # Try 20 starting nodes for robustness
    # N=50, this is very fast.
    for start in range(20):
        tour = _nearest_neighbor_tour(start, distance_matrix)
        tour = _two_opt(tour, distance_matrix)
        length = _tour_length(tour, distance_matrix)
        
        if length < best_len:
            best_len = length
            best_tour = tour
            
    if best_tour is not None:
        boost_factor = 10.0
        for i in range(n):
            u = best_tour[i]
            v = best_tour[(i + 1) % n]
            base_heuristic[u, v] *= boost_factor
            base_heuristic[v, u] *= boost_factor
            
    return base_heuristic
# EVOLVE-BLOCK-END
