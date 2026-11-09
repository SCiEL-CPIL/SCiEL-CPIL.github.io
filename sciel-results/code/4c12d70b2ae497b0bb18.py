"""Guided local search penalty for TSP: smooth sigmoid-weighted disruption."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 305.0  # Safety margin before 330s kill


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

    # For each node, compute the median of incident distances (excluding self)
    local_scale = np.median(d_work, axis=1)
    # Guard against zero
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
        # Power-law scaling for stronger differentiation
        usage_scaled = np.power(usage / u_max, 1.5)
        penalty += 0.12 * usage_scaled
    else:
        penalty += np.zeros_like(usage)

    # Smooth sigmoid-weighted disruption on over-long edges
    # Center at ratio=1.0, steep enough to give near-full weight to clearly long edges
    center = 1.0
    steepness = 5.0
    weight = 1.0 / (1.0 + np.exp(-steepness * (ratio - center)))

    # Penalty magnitude scales with how much the edge exceeds local scale
    excess_ratio = np.maximum(ratio - center, 0.0)
    mag = 0.30 * tour_len + 0.20 * mean_tour_edge + 0.40 * excess_ratio * mean_tour_edge

    # Apply smooth penalty to all tour edges, weighted by sigmoid
    for idx in range(n):
        a, b = i[idx], j[idx]
        p = weight[idx] * mag[idx]
        penalty[a, b] += p
        penalty[b, a] += p

    # Anti-clustering: penalize edges that are very short relative to local scale
    # These are "obvious" edges that 2-opt keeps re-adding; nudge exploration
    short_ratio = 1.0 / np.maximum(ratio, 1e-6)
    short_center = 1.5
    short_steepness = 4.0
    short_weight = 1.0 / (1.0 + np.exp(short_steepness * (short_ratio - short_center)))
    for idx in range(n):
        if short_weight[idx] > 0.15:
            a, b = i[idx], j[idx]
            p = 0.06 * short_weight[idx] * tour_len[idx]
            penalty[a, b] += p
            penalty[b, a] += p

    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty
    result = d + penalty

    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
