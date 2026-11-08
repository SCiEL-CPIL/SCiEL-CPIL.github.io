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
    n = len(local_opt_tour)
    
    # Compute tour length under true distances
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
    # margin = 0.15 means we add ~15% of tour length as total penalty
    margin = 0.15
    alpha = margin * tour_len / edge_sum
    
    # Clamp alpha to a reasonable range to avoid extreme perturbations
    alpha = np.clip(alpha, 0.0, 0.5)
    
    # Create penalty matrix: only penalize edges in the current tour
    penalty = np.zeros_like(edge_distance)
    for i in range(n):
        j = (i + 1) % n
        penalty[i, j] = alpha * edge_n_used[i, j]
        penalty[j, i] = alpha * edge_n_used[j, i]
    
    # Return penalized distances
    result = edge_distance + penalty
    
    # Add tiny random perturbation to break symmetry
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-6, size=(n, n))
    result += noise
    
    # Ensure symmetry
    result = (result + result.T) / 2.0
    
    # Zero the diagonal
    np.fill_diagonal(result, 0.0)
    
    # Ensure non-negativity and finiteness
    result = np.maximum(result, 0.0)
    result = np.where(np.isfinite(result), result, edge_distance)
    
    return result
# EVOLVE-BLOCK-END
