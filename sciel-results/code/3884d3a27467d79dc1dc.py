import numpy as np

def _greedy_tour(dist):
    n = dist.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        # Find nearest unvisited
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                if dist[last][j] < min_dist:
                    min_dist = dist[last][j]
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
    return tour

def _two_opt(tour, dist):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Reverse segment tour[i..j]
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1] if j+1<n else tour[0])
                # Note: tour is a cycle.
                # Let's define the cost change.
                # Edges removed: (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n])
                # Edges added: (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%n])
                
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                d = tour[(j+1) % n]
                
                # Check if i and j are adjacent or wrap around, 2-opt usually skips trivial cases
                if i == j or (j == i + 1) or (i == 1 and j == n - 1):
                    continue

                cost_removed = dist[a][b] + dist[c][d]
                cost_added = dist[a][c] + dist[b][d]
                
                if cost_added < cost_removed - 1e-10:
                    # Reverse tour[i..j]
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour

def heuristics(distance_matrix):
    dist = np.array(distance_matrix)
    n = dist.shape[0]
    
    # 1. Generate Greedy Tour
    tour = _greedy_tour(dist)
    
    # 2. Optimize with 2-Opt
    tour = _two_opt(tour, dist)
    
    # 3. Build Heuristic
    # Base heuristic: 1 / distance
    # Avoid division by zero
    eps = 1e-9
    base_heur = 1.0 / (dist + eps)
    
    # Boost edges in the 2-Opt tour
    # We want to make these edges very attractive.
    # Let's multiply by a factor, e.g., 10.
    # Or set to a very high value?
    # If we set to infinity, ants will always pick them if available.
    # This might restrict exploration too much.
    # Let's use a multiplier.
    
    boost_factor = 10.0
    for i in range(n):
        j = (i + 1) % n
        u = tour[i]
        v = tour[j]
        base_heur[u][v] *= boost_factor
        base_heur[v][u] *= boost_factor
        
    # Ensure positive
    base_heur[base_heur <= 0] = 1e-9
    
    return base_heur
