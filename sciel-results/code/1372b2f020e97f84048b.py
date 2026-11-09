"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(tour, dist):
    """Improve a tour using 2-opt."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # 2-opt: reverse segment tour[i+1..j]
                # Current edges: (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%n])
                # New edges: (tour[i], tour[j]) and (tour[i+1], tour[(j+1)%n])
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                if new_cost < old_cost - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour


def _nearest_neighbor_tour(dist, start=0):
    """Generate a tour using nearest neighbor heuristic."""
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    n = distance_matrix.shape[0]
    
    # Generate multiple reference tours and optimize them
    reference_tours = []
    
    # Nearest neighbor from node 0
    nn_tour = _nearest_neighbor_tour(distance_matrix, 0)
    optimized = _two_opt(nn_tour, distance_matrix)
    reference_tours.append(optimized)
    
    # Check time
    if time.time() > deadline:
        # Fallback: simple inverse distance
        return 1.0 / (distance_matrix + 1e-9)
    
    # Nearest neighbor from a few other starts
    for start in [1, n // 2]:
        if time.time() > deadline - 20:
            break
        nn_tour = _nearest_neighbor_tour(distance_matrix, start)
        optimized = _two_opt(nn_tour, distance_matrix)
        reference_tours.append(optimized)
    
    # Build edge attractiveness matrix
    # For each edge, count how many times it appears in reference tours
    # and weight by inverse tour length
    edge_attractiveness = np.ones((n, n))
    
    for tour in reference_tours:
        tour_len = sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
        weight = 1.0 / (tour_len + 1e-9)
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            # Add to both directions (but ACO uses directed edges)
            edge_attractiveness[u, v] += weight
            # Also add to reverse for symmetry in heuristic
            edge_attractiveness[v, u] += weight * 0.5
    
    # Normalize and apply power law to amplify differences
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(edge_attractiveness, 0.0)
    
    # Apply power law: higher values for more attractive edges
    min_val = np.min(edge_attractiveness[edge_attractiveness > 0])
    max_val = np.max(edge_attractiveness)
    
    if max_val > min_val:
        # Normalize to [0, 1] then apply power
        normalized = (edge_attractiveness - min_val) / (max_val - min_val + 1e-12)
        # Power > 1 amplifies differences
        powered = normalized ** 2.0
        # Scale to reasonable range
        edge_attractiveness = 1.0 + 10.0 * powered
    else:
        edge_attractiveness = np.ones((n, n))
        np.fill_diagonal(edge_attractiveness, 0.0)
    
    # Ensure no zero diagonal issues
    np.fill_diagonal(edge_attractiveness, 0.0)
    
    # Final check: ensure all off-diagonal entries are positive
    mask = ~np.eye(n, dtype=bool)
    edge_attractiveness[mask] = np.maximum(edge_attractiveness[mask], 1e-9)
    
    return edge_attractiveness
# EVOLVE-BLOCK-END
