"""Heuristic matrix for ant colony optimization of TSP."""
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    
    Strategy:
    1. Find a high-quality reference tour using Greedy NN + 2-Opt.
    2. Build a heuristic that strongly favors edges in the reference tour
       and moderately favors short edges not in the tour.
    3. This 'locks in' a good solution while still allowing ACO to explore.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Helper: compute tour length
    def tour_length(tour):
        total = 0.0
        for i in range(len(tour)):
            total += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return total
    
    # Step 1: Greedy Nearest Neighbor starting at node 0
    def nearest_neighbor_tour():
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        tour = [0]
        current = 0
        for _ in range(n - 1):
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = int(np.argmin(dists))
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        return tour
    
    # Step 2: 2-Opt improvement
    def two_opt(tour, max_iterations=1000):
        tour = list(tour)
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            for i in range(1, n - 1):
                # Check if reversing tour[i..j] improves
                for j in range(i + 1, n):
                    if j == (i + n - 1) % n:
                        continue
                    # Current edges: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    a, b = tour[i - 1], tour[i]
                    c, d = tour[j], tour[(j + 1) % n]
                    old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                    new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                    if new_cost < old_cost - 1e-10:
                        # Reverse the segment
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        improved = True
                        if time.time() - start_time > 280:
                            return tour
        return tour
    
    # Step 3: Find reference tour
    ref_tour = nearest_neighbor_tour()
    ref_tour = two_opt(ref_tour, max_iterations=2000)
    
    # Try a few random restarts with 2-opt to find even better tours
    best_tour = ref_tour
    best_len = tour_length(best_tour)
    
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() - start_time > 270:
            break
        # Random tour
        tour = list(range(n))
        rng.shuffle(tour)
        tour = two_opt(tour, max_iterations=1000)
        tl = tour_length(tour)
        if tl < best_len:
            best_len = tl
            best_tour = tour
    
    # Step 4: Build heuristic matrix
    # Strongly favor edges in the best tour, moderately favor short edges otherwise
    heuristic = np.ones((n, n), dtype=np.float64)
    
    # Mark edges in the best tour
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        heuristic[u, v] = 1e6  # Very strong preference
        heuristic[v, u] = 1e6  # Symmetric for safety (though ACO uses directed)
    
    # For non-tour edges, use inverse distance with a moderate boost
    # This helps ACO find alternative good edges if it needs to deviate
    for i in range(n):
        for j in range(n):
            if i != j and heuristic[i, j] < 1e5:
                d = distance_matrix[i, j]
                heuristic[i, j] = 1.0 / (d + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
