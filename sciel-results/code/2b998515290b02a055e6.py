import numpy as np

def _nearest_neighbor_tour(dist):
    """
    Generates a tour using the Nearest Neighbor heuristic.
    """
    n = len(dist)
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                d = dist[current, j]
                if d < min_dist:
                    min_dist = d
                    next_node = j
        if next_node == -1:
            break
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def _two_opt(tour, dist):
    """
    Improves a tour using the 2-opt heuristic.
    """
    n = len(tour)
    if n <= 3:
        return tour[:]
    
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += dist[t[i], t[(i + 1) % n]]
        return total
    
    best_tour = tour[:]
    best_len = tour_length(best_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Skip if the edges are adjacent in the tour structure in a way that doesn't form a valid 2-opt move
                # Specifically, we are swapping the direction of the segment i..j
                # Edges involved: (i-1, i) and (j, j+1)
                # New edges: (i-1, j) and (i, j+1)
                
                prev_i = tour[i - 1]
                curr_i = tour[i]
                curr_j = tour[j]
                next_j = tour[(j + 1) % n]
                
                # Calculate delta
                # If d2 < d1, the move improves the tour
                d_old = dist[prev_i, curr_i] + dist[curr_j, next_j]
                d_new = dist[prev_i, curr_j] + dist[curr_i, next_j]
                
                if d_new < d_old - 1e-10:
                    # Reverse the segment from i to j
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    # Break out of inner loop to restart the scan
                    # This is a standard strategy to ensure we don't miss interactions
                    break
            if improved:
                break
        
        if improved:
            new_len = tour_length(tour)
            if new_len < best_len:
                best_len = new_len
                best_tour = tour[:]
            # If new_len is not < best_len, it might be due to floating point or
            # the move didn't actually improve the global length (unlikely if local delta improved)
            # But we keep the best_tour found so far.
            
    return best_tour

def heuristics(distance_matrix):
    """
    Computes a heuristic matrix for Ant Colony Optimization.
    Uses a 2-opt improved nearest neighbor tour to create a strong structural prior.
    """
    dist = np.array(distance_matrix, dtype=float)
    n = dist.shape[0]
    
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[1.0]])
        
    # 1. Generate a high-quality initial tour
    nn_tour = _nearest_neighbor_tour(dist)
    opt_tour = _two_opt(nn_tour, dist)
    
    # 2. Base heuristic: inverse distance
    # This provides a geometric prior.
    safe_dist = dist + 1e-9
    H = 1.0 / safe_dist
    
    # 3. Boost edges that are part of the optimized tour
    # This guides the ACO ants to follow the structure of the good tour.
    # The boost factor determines the strength of this guidance.
    boost = 100.0
    
    for i in range(n):
        u = opt_tour[i]
        v = opt_tour[(i + 1) % n]
        H[u, v] *= boost
        H[v, u] *= boost
        
    return H
