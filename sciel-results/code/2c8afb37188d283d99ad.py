"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0  # seconds of safety margin


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _start_time)


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    GLS penalty: penalize edges in the current local optimum tour
    based on their historical usage frequency.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()

    result = edge_distance.copy()

    # Compute median edge distance for scale normalization
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0

    # Build penalty: only penalize edges in the current tour
    # Penalty grows with usage count, scaled by median distance
    # Use a smooth saturating function: penalty = alpha * med_dist * usage / (1 + usage)
    alpha = 0.25  # penalty strength

    # Identify edges in the current tour
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u < v:
            tour_edges.add((u, v))
        else:
            tour_edges.add((v, u))

    # Apply penalty to tour edges based on usage
    for (u, v) in tour_edges:
        usage = edge_n_used[u, v]
        if usage > 0:
            # Smooth penalty that increases with usage but has diminishing returns
            penalty = alpha * med_dist * usage / (1.0 + usage)
            result[u, v] += penalty
            result[v, u] += penalty

    # Small global perturbation to help escape local optima
    # Add tiny random noise to non-tour edges to break symmetry
    # Use deterministic pseudo-random based on matrix indices for reproducibility
    noise_scale = 0.01 * med_dist
    # Only add noise to a subset of edges to keep computation fast
    if _time_left() > 0.5:
        # Add small structured perturbation
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in tour_edges and (j, i) not in tour_edges:
                    # Deterministic small perturbation based on position
                    pert = noise_scale * 0.1 * np.sin(i * 0.7 + j * 1.3)
                    result[i, j] += pert
                    result[j, i] += pert

    return result
# EVOLVE-BLOCK-END
