"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    GLS penalty: penalize edges in the current local optimum tour
    based on their historical usage frequency.
    
    Combines:
    1. A mild global soft penalty proportional to edge usage (multiplicative).
    2. A stronger targeted penalty on edges in the current local-optimal tour,
       scaled by their usage count, to specifically break the current solution.
    3. A quadratic penalty for frequently used edges in the current tour.
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s kill

    n = len(local_opt_tour)
    if n <= 1:
        return edge_distance.copy()

    result = edge_distance.copy()

    # Scale penalty relative to typical edge distances
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0

    # Get max usage for normalization
    max_used = edge_n_used.max()
    if max_used <= 0:
        max_used = 1.0

    # Global soft penalty: apply a small penalty to all edges based on usage.
    # This helps the search escape by making previously-used edges less attractive.
    global_penalty = 1.0 + 0.01 * (edge_n_used / max_used)
    result *= global_penalty

    # Penalty per unit of usage count for targeted tour edges
    # Using a combination of multiplicative and additive penalties
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        used = edge_n_used[u, v]
        
        # Multiplicative penalty: base multiplier plus scaling with usage
        tour_penalty_mult = 1.30 + 0.40 * (used / max_used) + 0.02 * used
        tour_penalty_mult = min(tour_penalty_mult, 5.0)
        
        # Additive penalty: scales with usage count
        tour_penalty_add = 0.15 * med_dist * used
        
        result[u, v] *= tour_penalty_mult
        result[v, u] *= tour_penalty_mult
        
        result[u, v] += tour_penalty_add
        result[v, u] += tour_penalty_add

    # Check time
    if time.time() > deadline:
        return result

    # Additional refinement: apply a smooth penalty that encourages diversification
    # Use soft-max style weighting based on edge usage
    # This creates a smoother landscape for the local search
    if time.time() < deadline - 10.0:
        # Normalize usage counts
        normalized_used = edge_n_used / (1.0 + max_used)
        # Apply a quadratic penalty for frequently used edges
        quad_penalty = 0.05 * med_dist * normalized_used ** 2
        # Only apply to edges in the current tour
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            result[u, v] += quad_penalty[u, v]
            result[v, u] += quad_penalty[v, u]

    # Safety: ensure all values are finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=0.0)

    return result
# EVOLVE-BLOCK-END
