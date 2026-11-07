"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Penalizes edges based on cumulative usage with saturating normalization,
    plus a small immediate penalty on edges in the current local-opt tour
    to ensure escape from the current basin.
    """
    n = edge_distance.shape[0]
    
    # Saturating penalty: grows with usage but bounded by alpha
    alpha = 0.55
    penalty = alpha * edge_n_used / (1.0 + edge_n_used)
    
    # Small immediate penalty on edges in the current tour to guarantee perturbation
    immediate_scale = 0.08
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    penalty += immediate_scale * tour_mask
    
    # Multiply distance matrix by (1 + penalty) to penalize frequently-used edges
    return edge_distance * (1.0 + penalty)
# EVOLVE-BLOCK-END
