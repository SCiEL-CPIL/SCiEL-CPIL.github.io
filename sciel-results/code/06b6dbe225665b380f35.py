"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nearest_neighbor_tour(d, start):
    """Construct a tour using the Nearest Neighbor heuristic starting from 'start'."""
    n = len(d)
    visited = np.zeros(n, dtype=bool)
    tour = np.zeros(n, dtype=int)
    tour[0] = start
    visited[start] = True
    current = start
    for i in range(1, n):
        dists = d[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour[i] = next_node
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(d, tour, max_passes=20):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = tour.copy()
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d_node = tour[j + 1] if j + 1 < n else tour[0]
                delta = d[a, c] + d[b, d_node] - d[a, b] - d[c, d_node]
                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
        if not improved:
            break
    return tour

def _compute_heuristic(d, deadline):
    """
    Compute a heuristic matrix that strongly guides the ACO toward good tours.
    Uses multi-start NN+2-opt to find a good reference tour, then constructs
    a heuristic that heavily favors edges in that tour while maintaining
    inverse-distance baseline for other edges.
    """
    n = len(d)
    
    # Find a good tour using multiple starts
    best_tour = None
    best_len = np.inf
    
    # Use diverse starting points
    starts = list(range(0, n, max(1, n // 10)))
    if 0 not in starts:
        starts.insert(0, 0)
    
    for start in starts:
        if time.time() > deadline:
            break
        tour = _nearest_neighbor_tour(d, start)
        tour = _two_opt(d, tour, max_passes=10)
        # Calculate length
        idx = np.arange(n)
        length = np.sum(d[tour, np.roll(tour, -1)])
        if length < best_len:
            best_len = length
            best_tour = tour.copy()
    
    if best_tour is None:
        # Fallback: simple inverse distance
        return 1.0 / (d + 1e-9)
    
    # Build heuristic matrix
    # Strategy: strong multiplicative boost for edges in the reference tour
    # combined with inverse-distance baseline
    
    H = 1.0 / (d + 1e-9)
    
    # Boost edges in the best tour significantly
    # Use a large multiplier to dominate the initial pheromone (which starts at 1)
    # Since pheromone starts at 1, and heuristic * pheromone determines probability,
    # we want tour edges to have much higher heuristic values
    
    tour_idx = np.arange(n)
    u = best_tour[tour_idx]
    v = best_tour[np.roll(tour_idx, -1)]
    
    # Set high attractiveness for tour edges in both directions
    H[u, v] = 50.0
    H[v, u] = 50.0
    
    # Ensure all values are positive and finite
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO of TSP.
    Larger values indicate more attractive edges.
    """
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s margin for the 330s limit
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = len(d)
    
    # Validate input
    if d.shape != (n, n) or not np.all(np.isfinite(d)):
        return np.ones((n, n)) * 1e-9
    
    try:
        H = _compute_heuristic(d, deadline)
    except Exception:
        # Fallback to simple inverse distance
        H = 1.0 / (d + 1e-9)
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
    
    # Final validation
    if H.shape != (n, n) or not np.all(np.isfinite(H)):
        H = 1.0 / (d + 1e-9)
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
    
    return H
# EVOLVE-BLOCK-END
