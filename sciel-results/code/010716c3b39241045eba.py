"""Guided local search penalty rule for TSP with adaptive GLS penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Adaptive GLS penalty: penalize edges in the current tour proportionally
    to their usage count, scaled so the total penalty makes the current tour
    non-optimal under the penalized distances.
    """
    # Compute tour length under true distances
    n = len(local_opt_tour)
    tour_len = 0.0
    for i in range(n):
        j = (i + 1) % n
        tour_len += edge_distance[i, j]
    
    # Sum of edge_n_used for edges in the current tour
    edge_sum = 0.0
    for i in range(n):
        j = (i + 1) % n
        edge_sum += edge_n_used[i, j]
    
    if edge_sum <= 0:
        return edge_distance.copy()
    
    # Adaptive alpha: scale penalty so that the total added penalty
    # is a meaningful fraction of the tour length
    # margin = 0.1 means we add ~10% of tour length as total penalty
    alpha = 0.1 * tour_len / edge_sum
    
    # Create penalty matrix: only penalize edges in the current tour
    penalty = np.zeros_like(edge_distance)
    for i in range(n):
        j = (i + 1) % n
        penalty[i, j] = alpha * edge_n_used[i, j]
        penalty[j, i] = alpha * edge_n_used[j, i]
    
    # Return penalized distances
    return edge_distance + penalty
# EVOLVE-BLOCK-END
