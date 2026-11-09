"""Guided local search penalty rule for TSP with adaptive sparse penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Penalize edges in the current local-optimal tour to encourage escape.
    
    Strategy:
    - Add a penalty to each edge in the current tour.
    - The penalty magnitude is proportional to the average edge length in the tour,
      scaled by how often those edges have been used (edge_n_used).
    - This ensures the current tour becomes strictly suboptimal under penalized distances,
      forcing 2-opt to find a different local optimum.
    - Use a margin to ensure the penalty is strong enough to escape the basin.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Compute the current tour length using true distances
    tour_length = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_length += edge_distance[u, v]
    
    avg_edge_len = tour_length / n if n > 0 else 1.0

    # Create a copy of the distance matrix
    result = edge_distance.copy()

    # For each edge in the current tour, add a penalty
    # Penalty is proportional to average edge length and usage count
    # Use a margin factor to ensure the current tour becomes suboptimal
    margin = 0.05  # Small margin to ensure strict suboptimality
    base_penalty = avg_edge_len * (margin + 0.1)
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Penalty scales with usage: more used edges get larger penalties
        usage = edge_n_used[u, v]
        penalty = base_penalty * (1.0 + 0.5 * usage / (1.0 + usage))
        result[u, v] += penalty
        result[v, u] += penalty

    return result
# EVOLVE-BLOCK-END
