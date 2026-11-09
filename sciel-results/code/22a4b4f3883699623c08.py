"""Guided local search penalty rule for TSP using sparse targeted edge-breaking."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Apply a sparse, targeted penalty to break the current local optimum.
    
    Strategy:
    1. Identify edges in the current tour that have been used previously (edge_n_used > 0).
    2. Assign a very high penalty to ONLY these specific edges.
    3. Leave all other edges at their true Euclidean distances.
    
    This forces the subsequent 2-opt to replace the "stuck" edges with new ones,
    while preserving the geometric accuracy of the rest of the landscape.
    This is structurally different from global Lagrangian penalties which distort
    the entire distance matrix.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Start with true distances
    modified_distance = edge_distance.copy()
    
    # Determine the scale of the penalty.
    # It needs to be large enough to override the 2-opt improvement gain of keeping the edge.
    # A typical 2-opt improvement is small (fraction of avg edge length).
    # So a penalty of 5-10x the average edge length should be sufficient to force replacement.
    avg_dist = np.mean(edge_distance)
    penalty_value = 10.0 * avg_dist
    
    # Identify edges in the current tour that have been used before
    # We penalize them to force the tour to change structure at these points.
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Only penalize if this edge has been used in a previous round
        # edge_n_used is symmetric, so checking one direction is enough
        if edge_n_used[u, v] > 0:
            modified_distance[u, v] += penalty_value
            modified_distance[v, u] += penalty_value
            
    return modified_distance
# EVOLVE-BLOCK-END
