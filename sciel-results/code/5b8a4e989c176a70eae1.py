import numpy as np

def heuristics(distance_matrix):
    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # 1. Compute Nearest Neighbor Tour starting from 0
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        # Find unvisited node with min distance from current
        # d[current, :] gives distances to all
        # Mask visited
        dists = d[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # 2. Create Heuristic Matrix
    beta = 2.0
    alpha = 0.5 # Boost factor
    
    # Base heuristic: 1 / d^beta
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    d_safe = np.maximum(d_safe, 1e-9)
    eta = 1.0 / (d_safe ** beta)
    
    # 3. Boost NN edges
    # For each edge (u, v) in tour, boost eta[u, v] and eta[v, u]
    # The tour is closed, so we add edge from last to 0
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        # Boost both directions? The ACO moves in one direction, but the matrix is symmetric.
        # The probability uses eta[i, j].
        # If the tour is u->v, we boost eta[u, v].
        # Since the matrix is symmetric, eta[v, u] is the same value initially.
        # But if we boost only eta[u, v], it breaks symmetry.
        # The ACO is directed (travel direction only).
        # However, the distance matrix is symmetric.
        # Usually, heuristics are symmetric.
        # Let's boost both to maintain symmetry, or just the direction?
        # "each ant adds 1/length to pheromone[i, j] for each step i to j of its closed tour, in travel direction only."
        # This implies the pheromone matrix is NOT symmetric initially (it starts at 1, symmetric).
        # But as ants travel, tau[i, j] != tau[j, i] if ants go different ways.
        # The heuristic is computed ONCE.
        # It is safer to keep the heuristic symmetric.
        eta[u, v] *= (1.0 + alpha)
        eta[v, u] *= (1.0 + alpha)

    # 4. Cleanup
    np.fill_diagonal(eta, 0.0)
    eta = np.where(np.isfinite(eta), eta, 0.0)
    eta = np.maximum(eta, 0.0)
    
    return eta
