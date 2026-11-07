import numpy as np
import time

def _nearest_neighbor_tour(d):
    """
    Construct a tour using the nearest neighbor heuristic starting from node 0.
    """
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Get distances from current node to all unvisited nodes
        dists = d[current, :].copy()
        dists[visited] = np.inf
        
        # Find the nearest unvisited node
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def _two_opt_improve(tour, d, max_passes=50):
    """
    Improve a tour using the 2-opt heuristic.
    Returns the improved tour.
    """
    n = len(tour)
    # Map tour indices to node indices
    # tour is a list of node indices
    # We work with the list directly
    
    # Helper to calculate tour length
    def tour_length(t):
        length = 0.0
        for i in range(len(t)):
            length += d[t[i], t[(i + 1) % len(t)]]
        return length
    
    current_tour = tour.copy()
    current_length = tour_length(current_tour)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Skip if i and j are adjacent in the cycle
                if i == 0 and j == n - 1:
                    continue
                # Skip if i and j are adjacent
                if j == i + 1:
                    continue
                
                # Calculate delta length for 2-opt move
                # Edges involved: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # New edges: (i-1, j), (i, j+1)
                # Note: indices in tour list are 0 to n-1
                # Nodes:
                u = current_tour[(i - 1) % n]
                v = current_tour[i]
                w = current_tour[i + 1]
                x = current_tour[(j - 1) % n]
                y = current_tour[j]
                z = current_tour[(j + 1) % n]
                
                # Old length contribution
                old_len = d[u, v] + d[x, y]
                # New length contribution
                new_len = d[u, x] + d[v, y]
                
                if new_len < old_len - 1e-10:
                    # Reverse the segment between i and j
                    # The segment is current_tour[i:j+1]
                    # Reverse it
                    current_tour[i:j+1] = current_tour[i:j+1][::-1]
                    current_length += new_len - old_len
                    improved = True
    
    return current_tour

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger is more attractive).
    
    Strategy:
    1. Compute a high-quality reference tour using Nearest Neighbor + 2-opt.
    2. Create a base heuristic based on inverse squared distance.
    3. Boost the edges present in the reference tour to guide the ACO.
    """
    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # 1. Generate reference tour
    nn_tour = _nearest_neighbor_tour(d)
    ref_tour = _two_opt_improve(nn_tour, d, max_passes=20)
    
    # 2. Create base heuristic
    # Avoid division by zero on diagonal (though diagonal is usually 0 in TSP, 
    # and ants don't pick visited nodes, but let's be safe)
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    # Inverse squared distance emphasizes short edges
    base_heuristic = 1.0 / (d_safe ** 2)
    
    # 3. Boost edges in the reference tour
    # The reference tour is a cycle: 0 -> ... -> n-1 -> 0
    boost_factor = 10.0
    
    for i in range(n):
        u = ref_tour[i]
        v = ref_tour[(i + 1) % n]
        base_heuristic[u, v] *= boost_factor
        base_heuristic[v, u] *= boost_factor
    
    # Ensure all values are positive and finite
    # Replace any non-positive or non-finite values with a small epsilon
    base_heuristic = np.where(np.isfinite(base_heuristic) & (base_heuristic > 0), base_heuristic, 1e-9)
    
    return base_heuristic
