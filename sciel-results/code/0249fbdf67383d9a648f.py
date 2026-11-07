"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty: penalize ONLY the edges of the current local
    optimum tour, with a smooth, adaptive magnitude based on historical usage.
    This forces 2-opt to restructure the tour while keeping non-tour distances
    undistorted.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()

    result = edge_distance.copy()

    # Get the edges of the current tour
    tour_edges = []
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges.append((j, k))

    # Get usage counts for tour edges
    usage_counts = np.array([edge_n_used[i, j] for i, j in tour_edges])

    if usage_counts.max() == 0:
        # No history yet; apply a small fixed penalty to break symmetry
        for i, j in tour_edges:
            result[i, j] *= 1.1
            result[j, i] *= 1.1
        return result

    # Normalize usage to [0, 1]
    max_usage = usage_counts.max()
    norm_usage = usage_counts / max_usage

    # Smooth exponential saturation for penalty magnitude
    # Higher usage -> stronger penalty, but saturates smoothly
    # alpha controls max penalty multiplier (1.0 means double the distance)
    alpha = 0.8
    penalty_multiplier = 1.0 + alpha * (1.0 - np.exp(-2.0 * norm_usage))

    # Apply penalty only to tour edges
    for idx, (i, j) in enumerate(tour_edges):
        mult = penalty_multiplier[idx]
        result[i, j] *= mult
        result[j, i] *= mult

    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=edge_distance)

    return result
# EVOLVE-BLOCK-END
