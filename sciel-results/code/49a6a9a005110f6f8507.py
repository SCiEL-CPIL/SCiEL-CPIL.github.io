import numpy as np
import time
import math

def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    deadline = start_time + 290.0  # Safety margin
    
    # 1. Base Heuristic: Inverse Distance
    # This is the standard, robust baseline.
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
    
    # 2. Edge Popularity Aggregation via Multi-Start Local Search
    # We run multiple diverse tours to identify "consensus" good edges.
    # Unlike the current program which uses ONE tour, we use MANY.
    # This creates a "soft" boost for edges that are frequently part of good tours,
    # rather than a "hard" lock-in to a single tour's structure.
    
    edge_count = np.zeros((n, n), dtype=float)
    
    def tour_length(tour):
        return sum(distance_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    
    def two_opt(tour):
        """Standard 2-Opt improvement."""
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    if i == 0 and j == len(tour) - 1:
                        continue
                    # Calculate delta
                    u, v = tour[i], tour[i+1]
                    x, y = tour[j], tour[(j+1) % len(tour)]
                    # Edges (u,v) and (x,y) are removed
                    # Edges (u,x) and (v,y) are added
                    delta = distance_matrix[u, x] + distance_matrix[v, y] - distance_matrix[u, v] - distance_matrix[x, y]
                    if delta < -1e-10:
                        # Reverse segment i+1 to j
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        return tour
    
    def or_opt(tour):
        """Or-Opt: Move 1 or 2 cities to another position."""
        improved = True
        while improved:
            improved = False
            for i in range(len(tour)):
                # Try moving 1 city
                city = tour[i]
                prev = tour[(i-1) % len(tour)]
                next_c = tour[(i+1) % len(tour)]
                
                # Cost of removing city
                cost_remove = distance_matrix[prev, city] + distance_matrix[city, next_c] - distance_matrix[prev, next_c]
                
                for j in range(len(tour)):
                    if j == i or j == (i-1) % len(tour) or j == (i+1) % len(tour):
                        continue
                    prev_j = tour[(j-1) % len(tour)]
                    next_j = tour[(j+1) % len(tour)]
                    
                    # Cost of inserting after j
                    cost_insert = distance_matrix[prev_j, city] + distance_matrix[city, next_j] - distance_matrix[prev_j, next_j]
                    
                    delta = cost_insert - cost_remove
                    if delta < -1e-10:
                        # Perform move
                        # Remove city from i
                        tour.pop(i)
                        # Adjust j if i < j
                        if i < j:
                            j -= 1
                        # Insert after j
                        tour.insert(j+1, city)
                        improved = True
                        break
                if improved:
                    break
        return tour
    
    def nn_tour(start_node):
        """Nearest Neighbor tour starting from start_node."""
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

    # Run multiple diverse starts
    num_starts = 20
    for start in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Vary the start node to ensure diversity
        # Start at 0, 1, 2... but also randomize the order of NN choice slightly?
        # Simple approach: Start at different nodes.
        s = start % n
        tour = nn_tour(s)
        tour = two_opt(tour)
        tour = or_opt(tour)
        
        # Vote for edges in this tour
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            edge_count[u, v] += 1.0
            edge_count[v, u] += 1.0

    # 3. Combine Base Heuristic and Popularity
    # We want to boost edges that are frequently in good tours.
    # The base H is 1/d. 
    # Let's multiply H by a factor derived from edge_count.
    # If an edge appears in all 20 tours, it's very likely good.
    # If it appears in 0, it's likely bad (or just not used by these specific starts).
    
    # Normalize edge_count
    max_count = np.max(edge_count)
    if max_count > 0:
        # Popularity factor: 1.0 to (1.0 + max_boost)
        # We use a softer boost to avoid collapsing diversity too much.
        # Boost = 1 + (edge_count / max_count) * boost_strength
        boost_strength = 2.0
        popularity_factor = 1.0 + (edge_count / max_count) * boost_strength
        
        H = H * popularity_factor
    
    # 4. Final Sanity Checks
    # Ensure no zeros or negatives
    H = np.maximum(H, 1e-9)
    H[np.diag_indices_from(H)] = 1e-9 # Diagonal shouldn't be used, but keep valid
    
    return H
