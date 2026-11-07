import numpy as np

def _nn_tour(D):
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        # Find nearest unvisited node
        distances = D[current].copy()
        distances[visited] = np.inf
        next_node = np.argmin(distances)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(tour, D):
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%n])
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                d = tour[(j+1) % n]
                
                # Check if edges are valid (not self-loops, which they aren't in a tour)
                # Distance change
                old_dist = D[a, b] + D[c, d]
                new_dist = D[a, c] + D[b, d]
                
                if new_dist < old_dist - 1e-10:
                    # Reverse segment i..j
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix)
    
    # Base heuristic: inverse distance
    H = 1.0 / (D + 1e-9)
    
    # Generate a high-quality tour using NN + 2-Opt
    nn_tour = _nn_tour(D)
    opt_tour = _two_opt(nn_tour, D)
    
    # Create a mask for edges in the optimal tour
    # Note: The tour is a cycle, so we have edges (tour[i], tour[(i+1)%n])
    n = D.shape[0]
    mask = np.zeros_like(D, dtype=bool)
    for i in range(n):
        u = opt_tour[i]
        v = opt_tour[(i + 1) % n]
        mask[u, v] = True
        mask[v, u] = True
        
    # Boost the edges in the tour
    # A high boost ensures the ACO focuses on this region
    # If boost is too high, it might prevent exploration, but for TSP,
    # staying close to a good tour is usually beneficial.
    boost_factor = 10.0
    H = H * (1.0 + boost_factor * mask)
    
    # Ensure positivity (already handled by 1/D, but just in case)
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
