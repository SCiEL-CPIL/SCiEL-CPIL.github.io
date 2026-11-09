"""Guided local search penalty for TSP: adaptive graduated penalties on over-long tour edges."""
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    if time.perf_counter() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    n = edge_distance.shape[0]
    if n <= 2:
        return np.asarray(edge_distance, dtype=float).copy()

    # True distances (always input is the true matrix)
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
    mean_tour_len = np.mean(tour_len)

    # Local geometric reference: use the 2nd-smallest incident distance per node
    # as a robust "expected short edge" scale.
    diag_mask = np.eye(n, dtype=bool)
    d_work = d.copy()
    d_work[diag_mask] = np.inf

    # Two smallest per row via argsort (n=100, so this is fast)
    order = np.argsort(d_work, axis=1)
    min1 = d_work[np.arange(n), order[:, 0]]
    min2 = d_work[np.arange(n), order[:, 1]]
    local_scale = np.maximum(min1, min2)
    local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)

    # Excess length of each tour edge beyond local expectation
    # Normalize by the sum of endpoint scales for comparability
    excess = tour_len - 0.5 * (local_scale[i] + local_scale[j])
    excess = np.maximum(excess, 0.0)  # only penalize over-length edges

    # Graduated penalty: quadratic in excess, scaled to be meaningful in distance units
    # The penalty should be large enough to make 2-opt avoid these edges
    # Use mean_tour_len as the scale for the penalty magnitude
    penalty_mag = 0.3 * excess + 0.15 * (excess ** 2) / max(mean_tour_len, 1e-6)

    # Apply graduated penalties to all over-length tour edges (not just top-k)
    # This gives a smoother perturbation landscape
    penalty = np.zeros((n, n), dtype=float)
    for idx in range(n):
        a, b = i[idx], j[idx]
        mag = penalty_mag[idx]
        if mag > 1e-8:
            penalty[a, b] += mag
            penalty[b, a] += mag

    # Base usage-based penalty: stronger than before to encourage exploration
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
    else:
        usage_scaled = np.zeros_like(usage)
    # Scale usage penalty by mean tour length for consistency
    penalty += 0.1 * usage_scaled * mean_tour_len

    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty
    result = d + penalty
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
