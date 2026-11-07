import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit
    
    n = distance_matrix.shape[0]
    
    # 1. Compute standard inverse distance heuristic
    eta = 1.0 / (distance_matrix + 1e-9)
    
    # 2. Build greedy tour from node 0
    unvisited = set(range(1, n))
    current = 0
    path = [0]
    
    for _ in range(n - 1):
        best_next = None
        best_dist = float('inf')
        for u in unvisited:
            d = distance_matrix[current, u]
            if d < best_dist:
                best_dist = d
                best_next = u
        unvisited.remove(best_next)
        path.append(best_next)
        current = best_next
    
    # Close the tour
    path_closed = path + [0]
    
    # 3. Apply 2-opt local search to improve the tour
    def tour_length(p):
        total = 0.0
        for i in range(len(p) - 1):
            total += distance_matrix[p[i], p[i+1]]
        return total
    
    def two_opt(p):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(p) - 2):
                if time.time() > deadline:
                    return p
                for j in range(i + 1, len(p) - 1):
                    if time.time() > deadline:
                        return p
                    # Calculate improvement
                    a, b = p[i-1], p[i]
                    c, d = p[j], p[j+1]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-10:
                        # Reverse segment
                        p[i:j+1] = reversed(p[i:j+1])
                        improved = True
                        if time.time() > deadline:
                            return p
        return p
    
    # Work on a copy for 2-opt
    tour = list(path)
    # 2-opt on the open path (not including the closing edge to 0)
    # Actually, let's work with the full cycle
    # Convert to cycle representation: [0, ..., last] and edge last->0
    # For 2-opt on a cycle, we need to be careful
    
    # Let's do 2-opt on the path 0 -> p1 -> p2 -> ... -> p_{n-1} -> 0
    # Represent as list of n nodes: [0, p1, ..., p_{n-1}]
    # Edge from node i to node i+1, and edge from last to first (0)
    
    def cycle_length(t):
        total = 0.0
        for i in range(len(t)):
            total += distance_matrix[t[i], t[(i+1) % len(t)]]
        return total
    
    def two_opt_cycle(t):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(t) - 1):
                if time.time() > deadline:
                    return t
                for j in range(i + 1, len(t)):
                    if time.time() > deadline:
                        return t
                    # Edges: (t[i-1], t[i]) and (t[j], t[(j+1)%n])
                    # New edges: (t[i-1], t[j]) and (t[i], t[(j+1)%n])
                    a, b = t[i-1], t[i]
                    c, d = t[j], t[(j+1) % len(t)]
                    delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                    if delta < -1e-10:
                        # Reverse segment from i to j
                        t[i:j+1] = reversed(t[i:j+1])
                        improved = True
                        if time.time() > deadline:
                            return t
        return t
    
    tour = two_opt_cycle(tour)
    
    # 4. Boost edges in the improved tour
    # Use a boost factor that's strong enough to guide but not too dominant
    boost_factor = 2.0
    
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        eta[u, v] *= boost_factor
        eta[v, u] *= boost_factor  # symmetric boost
    
    # 5. Ensure no non-positive values (shouldn't happen with inverse distance)
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
