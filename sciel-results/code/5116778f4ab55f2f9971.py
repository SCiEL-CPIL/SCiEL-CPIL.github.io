import numpy as np
import time


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 250  # safety margin before 300s limit

    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    # Set diagonal to large value to avoid self-loops
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)

    # Strategy: Build a strong heuristic based on a combination of:
    # 1. Inverse distance (standard)
    # 2. Nearest neighbor tour edges (emphasize edges in a good tour)
    # 3. 2-opt improvement of that tour

    # Step 1: Find a good tour using Nearest Neighbor
    def nearest_neighbor_tour(dist, n):
        visited = [False] * n
        tour = [0]
        visited[0] = True
        current = 0
        for _ in range(n - 1):
            # Find nearest unvisited
            best_dist = float('inf')
            best_node = -1
            for j in range(n):
                if not visited[j] and dist[current, j] < best_dist:
                    best_dist = dist[current, j]
                    best_node = j
            if best_node == -1:
                break
            tour.append(best_node)
            visited[best_node] = True
            current = best_node
        return tour

    # Step 2: 2-opt improvement
    def two_opt(tour, dist, n, max_iterations=100):
        tour = list(tour)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if i == 0 and j == n - 1:
                        continue  # same edge
                    # Calculate delta
                    a, b = tour[i], tour[i + 1]
                    c, d_ = tour[j], tour[(j + 1) % n]
                    old_dist = dist[a, b] + dist[c, d_]
                    new_dist = dist[a, c] + dist[b, d_]
                    if new_dist < old_dist - 1e-10:
                        # Reverse segment between i+1 and j
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        improved = True
        return tour

    # Build nearest neighbor tour
    nn_tour = nearest_neighbor_tour(d_safe, n)

    # Apply 2-opt improvement
    improved_tour = two_opt(nn_tour, d_safe, n, max_iterations=50)

    # Create edge set from the improved tour
    tour_edges = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        u = improved_tour[i]
        v = improved_tour[(i + 1) % n]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0

    # Base heuristic: inverse distance with exponent
    alpha = 2.0
    base_eta = 1.0 / (d_safe ** alpha)

    # Boost edges that are in the good tour
    # Use a multiplicative boost for tour edges
    boost = 1.0 + 2.0 * tour_edges

    eta = base_eta * boost

    # Symmetrize
    eta = np.maximum(eta, eta.T)

    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

    # Set diagonal to small value
    np.fill_diagonal(eta, 1e-9)

    return eta
