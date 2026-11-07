"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: penalize edges in the current local
    optimal tour, scaled by a smooth saturating function of their usage count.
    
    Design:
    - Only penalize edges that are in the current tour (targeted)
    - Penalty magnitude grows with usage count using a soft saturating function
    - Scale penalty relative to the mean edge distance for robustness
    - Add a small global penalty for frequently-used edges to encourage diversity
    """
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill
    
    n = edge_distance.shape[0]
    result = edge_distance.copy()
    
    # Compute mean edge distance for scaling
    mean_dist = np.mean(edge_distance)
    
    # Identify edges in the current tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges[u, v] = True
        tour_edges[v, u] = True
    
    # Compute a smooth penalty factor based on usage count
    # Use a soft saturation: penalty grows with usage but saturates
    # penalty_factor = 1 + alpha * (edge_n_used / (1 + beta * edge_n_used))
    alpha = 0.5  # maximum additional penalty multiplier
    beta = 0.3   # saturation rate
    
    # Compute per-edge penalty for tour edges
    usage = edge_n_used.copy()
    penalty_factor = 1.0 + alpha * (usage / (1.0 + beta * usage))
    
    # Apply penalty only to tour edges
    # Scale by mean distance to make penalty meaningful relative to edge lengths
    penalty_scale = mean_dist * 0.1
    
    # Add targeted penalty to tour edges
    result[tour_edges] += penalty_factor[tour_edges] * penalty_scale
    
    # Add a small global penalty for frequently-used edges (not just tour edges)
    # This encourages the search to avoid previously-used edges even if not in current tour
    global_penalty_weight = 0.05
    global_penalty = global_penalty_weight * mean_dist * (usage / (1.0 + 0.5 * usage))
    result += global_penalty
    
    # Ensure the result is finite and non-negative
    result = np.maximum(result, 0.0)
    result = np.where(np.isfinite(result), result, edge_distance)
    
    # Symmetrize to be safe
    result = (result + result.T) / 2.0
    
    # Time guard: if we're close to deadline, return immediately
    if time.time() - t0 > DEADLINE:
        return result
    
    return result
# EVOLVE-BLOCK-END
