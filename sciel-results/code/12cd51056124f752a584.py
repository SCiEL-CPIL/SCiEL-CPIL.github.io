"""Guided local search penalty rule for TSP with adaptive tour-edge penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Penalize edges of the current local optimum with an adaptive margin
    that ensures the current tour becomes strictly suboptimal under
    penalized distances, forcing 2-opt to escape the basin.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()

    # Compute current tour length under true distances
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))

    tour_len = sum(edge_distance[u, v] for u, v in tour_edges)
    if tour_len <= 0:
        return edge_distance.copy()

    # Average edge length
    avg_edge = tour_len / n

    # Adaptive penalty: ensure each tour edge gets enough penalty
    # to make the total tour length increase by a margin (e.g., 5% of tour length)
    margin = 0.05 * tour_len
    base_penalty = margin / n  # per-edge penalty to achieve total margin

    # Scale penalty by usage: edges used more get more penalty
    # Use a smooth scaling: penalty * (1 + usage) / (1 + max_usage)
    max_usage = edge_n_used.max()
    if max_usage <= 0:
        scale_factor = 1.0
    else:
        scale_factor = 1.0

    # Build penalized matrix
    result = edge_distance.copy()

    # Apply penalty to tour edges
    for u, v in tour_edges:
        usage = edge_n_used[u, v]
        # Adaptive penalty: base + usage-dependent component
        # The usage-dependent component uses a soft scaling to avoid discontinuities
        penalty = base_penalty * (1.0 + 0.5 * usage / (1.0 + max_usage))
        result[u, v] += penalty
        result[v, u] += penalty

    return result
# EVOLVE-BLOCK-END
