"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mst_edges(distance_matrix):
    """Compute MST edges using Prim's algorithm."""
    n = distance_matrix.shape[0]
    in_mst = [False] * n
    min_dist = [np.inf] * n
    parent = [-1] * n
    
    min_dist[0] = 0.0
    mst_edges = []
    
    for _ in range(n):
        # Find the vertex with the minimum distance not yet in MST
        u = -1
        best_dist = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best_dist:
                best_dist = min_dist[i]
                u = i
        
        if u == -1:
            break # Disconnected graph, shouldn't happen for TSP
            
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))
            
        # Update min_dist and parent for neighbors
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
                
    return mst_edges

def _greedy_tour(distance_matrix, start_node=0):
    """Generate a greedy nearest-neighbor tour."""
    n = distance_matrix.shape[0]
    visited = [False] * n
    tour = [start_node]
    visited[start_node] = True
    current = start_node
    
    for _ in range(n - 1):
        nearest = -1
        min_d = np.inf
        for v in range(n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < min_d:
                    min_d = d
                    nearest = v
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
        
    # Close the tour
    tour.append(start_node)
    return tour

def _two_opt(tour, distance_matrix, max_iterations=1000, time_limit=5.0):
    """Optimize tour using 2-Opt."""
    start_time = time.time()
    n = len(tour) - 1 # Number of cities
    # tour is a list of nodes, tour[0] == tour[-1]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(1, n - 1):
            if (time.time() - start_time) > time_limit:
                return tour
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_limit:
                    return tour
                
                # Calculate delta length
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[j+1])
                # Note: tour is 0-indexed, tour[k] is the k-th node in the path
                # The segment being reversed is tour[i] ... tour[j]
                
                d1 = distance_matrix[tour[i-1], tour[i]]
                d2 = distance_matrix[tour[j], tour[j+1]]
                d3 = distance_matrix[tour[i-1], tour[j]]
                d4 = distance_matrix[tour[i], tour[j+1]]
                
                if d3 + d4 < d1 + d2 - 1e-10:
                    # Reverse the segment tour[i] ... tour[j]
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    
    return tour

def heuristics(distance_matrix):
    # Ensure input is a numpy array
    dist = np.asarray(distance_matrix, dtype=float)
    n = dist.shape[0]
    
    # Initialize heuristic with inverse distance
    # Avoid division by zero
    heuristic = 1.0 / (dist + 1e-9)
    
    # 1. Boost MST edges
    mst_edges = _compute_mst_edges(dist)
    for u, v in mst_edges:
        heuristic[u, v] *= 3.0
        heuristic[v, u] *= 3.0
        
    # 2. Boost 2-Opt optimized greedy tour edges
    greedy_tour = _greedy_tour(dist, start_node=0)
    optimized_tour = _two_opt(greedy_tour, dist, max_iterations=500, time_limit=2.0)
    
    for i in range(len(optimized_tour) - 1):
        u = optimized_tour[i]
        v = optimized_tour[i+1]
        heuristic[u, v] *= 8.0
        heuristic[v, u] *= 8.0
        
    # Ensure positivity and finiteness
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
