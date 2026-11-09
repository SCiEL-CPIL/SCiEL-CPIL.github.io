import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # 1. Generate a high-quality reference tour using Greedy NN + 2-Opt
    # Greedy Nearest Neighbor starting at 0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited neighbor
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    # 2-Opt improvement
    def tour_length(tour):
        return sum(distance_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                if i == 0 and j == len(tour) - 1:
                    continue
                # Reverse tour[i+1..j]
                new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                if tour_length(new_tour) < tour_length(tour) - 1e-10:
                    tour = new_tour
                    improved = True
    
    # 2. Construct heuristic matrix based on the reference tour
    # We want edges in the good tour to have high attractiveness.
    # We can use a decay or simply assign high values to tour edges.
    # A common technique is to set heuristic[i,j] = 1/d(i,j) for all, 
    # but boost tour edges significantly or use the tour structure to guide.
    # Here, we set heuristic to 1/d for all, but multiply tour edges by a large factor
    # or simply define heuristic based on the tour edges directly.
    # Let's try: heuristic[i,j] = 1 if (i,j) is in tour, else 1/d(i,j) * small_factor?
    # Or better: Use the tour to define a "potential".
    # Let's stick to a robust static heuristic derived from the tour:
    # Set heuristic[i,j] = 1.0 / (distance_matrix[i,j] + 1e-9)
    # But for edges in the 2-opt tour, set them to a very high value, e.g., 10.0?
    # Actually, the ant algorithm multiplies pheromone * heuristic.
    # If we make heuristic huge for tour edges, ants will almost always pick them.
    # This might collapse diversity too early.
    
    # Alternative: Use the tour edges to define a base, and use inverse distance for others.
    # Let's try a hybrid: 
    # H[i,j] = 1/d(i,j) for all i!=j
    # Then, for each edge (u,v) in the reference tour, H[u,v] *= 5.0 (or some boost)
    
    H = 1.0 / (distance_matrix + 1e-9)
    H[np.diag_indices_from(H)] = 0.0  # Avoid self-loops in heuristic if needed, though ants check visited
    
    # Boost edges in the reference tour
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        H[u, v] *= 10.0
        H[v, u] *= 10.0  # Distance matrix is symmetric, but let's be safe
        
    # Ensure positive values
    H = np.maximum(H, 1e-9)
    
    return H
