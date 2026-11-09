"""Guided local search penalty for TSP: topological disruption via shortest-edge removal."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 310.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    n = edge_distance.shape[0]
    if n <= 2:
        return np.asarray(edge_distance, dtype=float).copy()

    # True distances
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]
    mean_tour_edge = np.mean(tour_len)

    # Compute local scale: for each node, the median of its incident distances
    diag_mask = np.eye(n, dtype=bool)
    d_work = d.copy()
    d_work[diag_mask] = np.inf

    local_scale = np.median(d_work, axis=1)
    local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)

    # Over-length ratio for each tour edge
    ratio = tour_len / (local_scale[i] + local_scale[j])

    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=float)

    # Usage-based penalty: penalize edges that have been used many times
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
        penalty += 0.15 * usage_scaled
    else:
        penalty += np.zeros_like(usage)

    # Topological disruption: identify the k SHORTEST edges in the current tour
    # and set their distance to a large constant to force 2-opt to restructure
    k = max(1, int(round(0.25 * n)))  # 25% of edges
    shortest_idx = np.argsort(tour_len)[:k]

    # Large penalty to effectively "remove" these edges from consideration
    large_penalty = 5.0 * mean_tour_edge
    for idx in shortest_idx:
        a, b = i[idx], j[idx]
        penalty[a, b] += large_penalty
        penalty[b, a] += large_penalty

    # Additional disruption: penalize edges in the tour that are significantly
    # longer than average (to break long-range connections)
    excess = tour_len - mean_tour_edge
    for idx in range(n):
        if excess[idx] > 0.3 * mean_tour_edge:
            a, b = i[idx], j[idx]
            mag = 0.15 * excess[idx]
            penalty[a, b] += mag
            penalty[b, a] += mag

    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty
    result = d + penalty

    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
