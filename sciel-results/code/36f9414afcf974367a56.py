# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    Larger values = more attractive.
    """
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(np.float64)
    
    # Base: inverse distance (avoid zero)
    base = 1.0 / (d + 1e-9)
    
    # --- Greedy nearest-neighbor tour with capacity ---
    # Simulate a greedy tour to identify "good" edges
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining = capacity
    tour_edges = np.zeros((n, n), dtype=np.float64)
    
    unvisited = n - 1
    while unvisited > 0:
        # Find best next customer
        best_j = -1
        best_score = -np.inf
        for j in range(1, n):
            if visited[j]:
                continue
            if demands[j] > remaining:
                continue
            # Score: prefer close, but also consider if this leads to bad future
            score = 1.0 / (d[current, j] + 1e-9)
            if best_score < score:
                best_score = score
                best_j = j
        if best_j == -1:
            # Return to depot
            current = 0
            remaining = capacity
            continue
        # Take edge current -> best_j
        tour_edges[current, best_j] += 1.0
        visited[best_j] = True
        unvisited -= 1
        remaining -= demands[best_j]
        current = best_j
        if remaining <= 0 or unvisited == 0:
            # Return to depot
            current = 0
            remaining = capacity
    
    # Normalize tour edge indicator
    tour_bonus = tour_edges / (tour_edges.sum() + 1e-9)
    
    # --- Depot proximity bonus ---
    # Edges from customers to depot are valuable for closing routes
    depot_dist = d[:, 0]  # distance from each node to depot
    depot_bonus = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        depot_bonus[i, 0] = 1.0 / (depot_dist[i] + 1e-9)
    # Also, edges from depot to customers that are close are good starts
    for j in range(1, n):
        depot_bonus[0, j] = 1.0 / (d[0, j] + 1e-9)
    
    # --- Capacity-aware penalty ---
    # Penalize edges between two high-demand customers if their combined demand
    # is a large fraction of capacity (wastes a vehicle)
    cap_penalty = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            combined = demands[i] + demands[j]
            if combined > capacity * 0.8:
                cap_penalty[i, j] = 1.0
            elif combined > capacity * 0.6:
                cap_penalty[i, j] = 0.5
    
    # --- Combine ---
    # Weights: base inverse distance, tour membership, depot proximity, capacity penalty
    w_base = 1.0
    w_tour = 2.0
    w_depot = 1.5
    w_cap = 0.5
    
    heuristic = (
        w_base * base +
        w_tour * tour_bonus * n +  # scale tour bonus to be comparable
        w_depot * depot_bonus +
        -w_cap * cap_penalty
    )
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Symmetrize? No, ACO uses directed edges. Keep as is.
    
    # Validate shape and finiteness
    if heuristic.shape != distance_matrix.shape:
        raise ValueError("Shape mismatch")
    if not np.all(np.isfinite(heuristic)):
        raise ValueError("Non-finite values in heuristic")
    
    return heuristic
# EVOLVE-BLOCK-END
