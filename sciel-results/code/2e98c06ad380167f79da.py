# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_DEADLINE = 300.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    """
    # Deadline guard
    if time.monotonic() - _START > _DEADLINE - 10.0:
        return edge_distance.copy()

    new_dist = edge_distance.copy()
    n = len(local_opt_tour)
    if n < 3:
        return new_dist

    alpha = 0.05
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        count = edge_n_used[u, v]
        if count > 0:
            penalty = alpha * count
            new_dist[u, v] += penalty
            new_dist[v, u] += penalty

    return new_dist
# EVOLVE-BLOCK-END
