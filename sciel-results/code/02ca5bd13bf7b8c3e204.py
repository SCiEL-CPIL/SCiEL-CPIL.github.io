"""Guided local search penalty rule for TSP using Lagrangian relaxation on edge usage."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Apply a Lagrangian penalty based on global edge usage counts.
    
    Instead of arbitrarily penalizing specific edges in the current tour, this approach
    adds a penalty proportional to the number of times each edge has been used in the
    history of the search. This creates a smooth, global pressure to explore new edges
    and escape the current basin of attraction, which is more robust than binary
    thresholding that might be too aggressive or too specific.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Ensure we have a working copy to avoid modifying the input
    modified_distance = edge_distance.copy()
    
    # Calculate the penalty scale.
    # The penalty should be significant enough to influence 2-opt decisions but not so
    # large that it completely distorts the geometry.
    # A common heuristic is to scale by the average edge length in the current tour
    # or the global average distance.
    
    # Calculate average edge length in the current tour to normalize the penalty
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
        
    if len(tour_edges) > 0:
        avg_tour_len = np.mean([edge_distance[u, v] for u, v in tour_edges])
    else:
        avg_tour_len = 1.0

    # The multiplier lambda determines the strength of the penalty.
    # We want the penalty for an edge used k times to be roughly k * lambda * avg_len.
    # If an edge is used 5 times, we want it to be significantly more expensive.
    # Let's try a moderate multiplier.
    # The usage counts are integers. We want the penalty to be comparable to the distance differences.
    # A multiplier of 0.1 * avg_len per use seems reasonable.
    # So penalty = 0.1 * avg_tour_len * edge_n_used
    
    # However, edge_n_used can be large. We should be careful not to overflow or make the matrix
    # dominated by penalties.
    
    # Let's use a slightly more adaptive multiplier.
    # If the max usage count is high, we might want a smaller multiplier to keep the scale sane.
    # But usually, in 20 iterations, usage counts are small (0-20).
    
    # Let's stick to a fixed multiplier relative to the average tour length.
    multiplier = 0.15 * avg_tour_len
    
    # Apply the penalty: d'_{uv} = d_{uv} + multiplier * n_{uv}
    # This is a vectorized operation for efficiency.
    modified_distance += multiplier * edge_n_used
    
    # Ensure symmetry (though edge_n_used is symmetric, it's good practice)
    # The addition preserves symmetry if edge_n_used is symmetric.
    
    return modified_distance
# EVOLVE-BLOCK-END
