"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Strategy:
    - Strongly penalize edges in the current local optimum tour to force escape
    - Use accumulated usage history for additional historical penalty
    - Apply exponential saturation for smooth, bounded penalty scaling
    - Only penalize edges that are actually in the current tour (not all used edges)
      This keeps non-tour distances undistorted, forcing 2-opt to restructure
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return edge_distance.copy()
    
    # Create a copy to work with
    result = edge_distance.copy()
    
    # Build the set of edges in the current tour
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Get usage counts for tour edges
    max_usage = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_usage == 0:
        max_usage = 1.0
    
    # Apply penalty to tour edges only
    # Base penalty: strong enough to force escape from current basin
    # Historical component: additional penalty based on how often edge was used
    for (u, v) in tour_edges:
        usage = edge_n_used[u, v]
        # Exponential saturation: smooth, bounded penalty
        # Scale so that penalty is significant but not overwhelming
        base_penalty = 0.5
        hist_penalty = 0.3 * (usage / (1.0 + max_usage))
        total_penalty = 1.0 + base_penalty + hist_penalty
        result[u, v] *= total_penalty
        result[v, u] *= total_penalty
    
    return result
# EVOLVE-BLOCK-END
