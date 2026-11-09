"""Rewritten program for guided local search penalty rule for TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    Returns a modified distance matrix that penalizes frequently used edges,
    encouraging the subsequent 2-opt to explore different tour structures.
    
    Uses a 2-opt move regret model: for each edge in the current tour,
    compute the minimal length increase to remove it via a single 2-opt swap,
    and penalize edges with low regret (easy to break) more heavily.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit
    
    n = edge_distance.shape[0]
    tour = np.asarray(local_opt_tour)
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Build list of tour edges
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        d = edge_distance[u, v]
        tour_edges.append((d, u, v))
    
    # For each tour edge, compute the 2-opt regret:
    # The minimal increase in tour length if we remove this edge via a 2-opt swap.
    # A 2-opt swap on edge (u, v) in the tour means:
    #   - Remove edge (u, v) and edge (x, y)
    #   - Add edges (u, x) and (v, y) or (u, y) and (v, x)
    # We want the minimal increase over all possible 2-opt moves that remove (u, v).
    
    # For efficiency, we compute regret for each edge in the tour.
    # Edge at position i connects tour[i] to tour[(i+1) % n].
    # To remove this edge via 2-opt, we need to find another edge (tour[j], tour[(j+1)%n])
    # such that the swap reduces the total length the least (i.e., minimal increase).
    
    # The 2-opt move that removes edge i and edge j (i != j) has change:
    #   delta = d(tour[i], tour[j]) + d(tour[i+1], tour[j+1]) - d(tour[i], tour[i+1]) - d(tour[j], tour[j+1])
    # or
    #   delta = d(tour[i], tour[j+1]) + d(tour[i+1], tour[j]) - d(tour[i], tour[i+1]) - d(tour[j], tour[j+1])
    # We take the minimum of these two for each pair (i, j).
    
    # For each edge i, regret[i] = min over j != i of min(delta1, delta2)
    # If all deltas are negative, regret is 0 (the edge is not in a 2-opt local optimum).
    # But since the tour is 2-opt optimal, all deltas should be >= 0.
    
    # To compute this efficiently, we iterate over all pairs. For n=100, that's ~5000 pairs.
    regrets = np.zeros(n)
    
    for i in range(n):
        u1 = tour[i]
        v1 = tour[(i + 1) % n]
        d1 = edge_distance[u1, v1]
        
        min_delta = np.inf
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue
            u2 = tour[j]
            v2 = tour[(j + 1) % n]
            d2 = edge_distance[u2, v2]
            
            # Two possible 2-opt swaps
            delta_a = edge_distance[u1, u2] + edge_distance[v1, v2] - d1 - d2
            delta_b = edge_distance[u1, v2] + edge_distance[v1, u2] - d1 - d2
            
            delta = min(delta_a, delta_b)
            if delta < min_delta:
                min_delta = delta
        
        # Regret is the minimal increase; if negative, it's 0 (shouldn't happen for 2-opt optimal)
        regrets[i] = max(0.0, min_delta)
    
    # Normalize regrets
    max_regret = np.max(regrets) if np.max(regrets) > 0 else 1.0
    regret_norm = regrets / (max_regret + 1e-10)
    
    # Create penalty matrix
    penalty = np.zeros_like(edge_distance)
    
    # Global usage-based penalty with smooth saturation
    alpha_global = 0.15
    penalty_global = alpha_global * (edge_n_used / (1.0 + edge_n_used))
    penalty += penalty_global
    
    # Local penalty on current tour edges:
    # Edges with low regret (easy to break) get MORE penalty
    # Edges with high regret (hard to break) get LESS penalty
    alpha_local_base = 0.6
    for idx, (d, u, v) in enumerate(tour_edges):
        # Inverse regret weighting: low regret -> high penalty
        weight = 1.0 - regret_norm[idx]
        # Add a minimum penalty
        local_penalty = alpha_local_base * (0.2 + 0.8 * weight)
        penalty[u, v] += local_penalty
        penalty[v, u] += local_penalty
    
    # Apply penalty to the distance matrix
    modified_distance = edge_distance + penalty
    
    # Ensure symmetry
    modified_distance = (modified_distance + modified_distance.T) / 2.0
    
    # Ensure finite values
    modified_distance = np.nan_to_num(modified_distance, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Safety: check elapsed time
    if time.time() > deadline:
        # Return a simple perturbation if we're running out of time
        return edge_distance + 0.1 * (edge_n_used / (1.0 + edge_n_used))
    
    return modified_distance
# EVOLVE-BLOCK-END
