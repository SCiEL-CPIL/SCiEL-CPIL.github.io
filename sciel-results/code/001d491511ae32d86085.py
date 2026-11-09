import numpy as np
import time

def _two_opt(tour, distance_matrix, time_limit):
    """Improve a tour using 2-opt. Returns the improved tour."""
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                n1, n2 = tour[i], tour[i + 1]
                n3, n4 = tour[j], tour[(j + 1) % n]
                
                d_old = distance_matrix[n1, n2] + distance_matrix[n3, n4]
                d_new = distance_matrix[n1, n3] + distance_matrix[n2, n4]
                
                if d_new < d_old - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
            if time.time() > time_limit:
                return tour
    return tour

def _nearest_neighbor_tour(distance_matrix, start_node):
    """Generate a tour using nearest neighbor heuristic."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start_node] = True
    tour = [start_node]
    current = start_node
    
    for _ in range(n - 1):
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Time limit for the entire heuristic computation
    # Leave some margin before the 330s wall-clock limit
    time_limit = start_time + 300
    
    # 1. Generate a high-quality reference tour using Greedy NN + 2-Opt
    # Start from node 0 as per the task description
    tour = _nearest_neighbor_tour(distance_matrix, 0)
    
    # 2-Opt improvement
    tour = _two_opt(tour, distance_matrix, time_limit)
    
    # 2. Build Heuristic Matrix
    # Base: 1 / d^2
    # Avoid division by zero
    safe_dist = distance_matrix + 1e-9
    H = 1.0 / (safe_dist ** 2)
    
    # Boost edges in the reference tour
    # The tour is a cycle: tour[0]-tour[1]-...-tour[n-1]-tour[0]
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        H[u, v] *= 2.0
        H[v, u] *= 2.0
        
    # Ensure positive
    H[H <= 0] = 1e-9
    
    return H
