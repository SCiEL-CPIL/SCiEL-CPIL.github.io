"""Guided local search penalty rule for TSP with sparse tour-edge penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Sparse GLS penalty: add a constant penalty to edges in the current tour.
    This forces the local search to escape the current basin by modifying
    the cost landscape only where the search is stuck.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    penalized = edge_distance.copy()

    # Identify edges in the current tour
    # local_opt_tour is a permutation of 0..n-1
    # Edges are (tour[i], tour[i+1]) for i in 0..n-1 (with wrap-around)
    
    # We add a penalty to these specific edges.
    # The penalty magnitude should be significant enough to break the local optimum
    # but not so large as to make the problem trivial or the matrix invalid.
    # A common choice in GLS is to add a penalty proportional to the average edge length
    # or a fixed fraction of the max distance.
    
    # Let's use a penalty that is a small fraction of the maximum distance in the matrix,
    # ensuring it's visible but doesn't overwhelm the geometry.
    # Alternatively, a fixed penalty like 0.1 or 0.05 times the mean edge length.
    
    max_dist = np.max(edge_distance)
    if max_dist == 0:
        return penalized
    
    # Penalty value: 5% of the max distance seems reasonable for 100 points in unit square.
    # Max distance is ~1.4. 5% is ~0.07.
    # Let's try a slightly stronger penalty to ensure escape: 10% of max distance.
    penalty = 0.1 * max_dist

    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        penalized[u, v] += penalty
        penalized[v, u] += penalty

    return penalized
# EVOLVE-BLOCK-END
