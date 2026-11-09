import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 250  # Leave 80s safety margin before 330s kill
    n = distance_matrix.shape[0]
    
    # 1. Generate a high-quality reference tour using Greedy NN + 2-Opt
    # Greedy Nearest Neighbor starting at 0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    # 2-Opt improvement
    def tour_length(t):
        length = 0
        for i in range(len(t)):
            u = t[i]
            v = t[(i + 1) % len(t)]
            length += distance_matrix[u, v]
        return length
        
    best_tour = list(tour)
    best_len = tour_length(best_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour)):
                if i == 0 and j == len(tour) - 1:
                    continue
                n1 = tour[i]
                n2 = tour[i + 1]
                n3 = tour[j]
                n4 = tour[(j + 1) % len(tour)]
                
                d_old = distance_matrix[n1, n2] + distance_matrix[n3, n4]
                d_new = distance_matrix[n1, n3] + distance_matrix[n2, n4]
                
                if d_new < d_old - 1e-10:
                    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    # 2. Build Heuristic Matrix
    # Base: 1 / d^2
    safe_dist = distance_matrix + 1e-9
    H = 1.0 / (safe_dist ** 2)
    
    # Boost edges in the reference tour
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        H[u, v] *= 2.0
        H[v, u] *= 2.0
        
    # Ensure positive
    H[H <= 0] = 1e-9
    
    return H
