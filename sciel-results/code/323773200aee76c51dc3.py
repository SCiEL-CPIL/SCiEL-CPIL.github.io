"""Guided local search penalty for TSP: topological disruption of over-long tour edges."""
import time
import numpy as np

_DEADLINE = time.perf_counter() + 310.0  # safety margin before the 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    if n <= 2:
        return edge_distance.copy()

    # True distances (always input is the true matrix)
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    # Ensure length == n and is a permutation; fallback to identity if malformed
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    # closing edge
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]

    # Local geometric reference: for each edge (a,b), the cheapest edge from a
    # and from b (excluding the edge itself) gives a "should be about this long" scale.
    # Use the 2nd-smallest incident distance as a robust local scale.
    diag_mask = np.eye(n, dtype=bool)
    d_work = d.copy()
    d_work[diag_mask] = np.inf

    # two smallest per row
    order = np.argsort(d_work, axis=1)
    min1 = d_work[np.arange(n), order[:, 0]]
    min2 = d_work[np.arange(n), order[:, 1]]
    local_scale = np.maximum(min1, min2)  # a robust "expected short edge" scale
    # Guard against zero scale
    local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)

    # Over-length ratio for each tour edge: how much longer than local expectation
    ratio = tour_len / (local_scale[i] + local_scale[j])  # normalize by both endpoints
    # Larger ratio => more "wasteful" edge that 2-opt tends to keep if neighbors are fixed.

    # We want to disrupt edges with the largest ratios (the structurally bad ones).
    # Rank the tour edges by ratio and remove the top fraction from the feasible set.
    k = max(1, int(round(0.25 * n)))  # top 25% of tour edges by over-length
    # indices of the k largest ratios
    top_idx = np.argsort(ratio)[-k:]

    # Build a penalty matrix (additive), zero by default
    penalty = np.zeros((n, n), dtype=float)

    # Base usage-based mild penalty for all edges seen before (symmetric)
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    # Scale usage penalty gently: proportional to log(1+usage) to avoid blowup
    u_max = usage.max()
    if u_max > 0:
        usage_scaled = np.log1p(usage) / np.log1p(u_max)  # in [0,1]
    else:
        usage_scaled = np.zeros_like(usage)
    # Add a small uniform-ish penalty weighted by usage (symmetric)
    penalty += 0.05 * usage_scaled

    # Topological disruption: set the worst tour edges to a large finite constant,
    # effectively removing them from the feasible set so 2-opt must restructure
    # the tour around alternative connections rather than just swapping slightly
    # longer edges. Use a large value relative to the tour's total length.
    tour_total = float(np.sum(tour_len))
    large_const = 10.0 * tour_total  # large enough to be effectively infeasible
    for idx in top_idx:
        a, b = i[idx], j[idx]
        penalty[a, b] += (large_const - d[a, b])
        penalty[b, a] += (large_const - d[b, a])

    # Symmetrize to be safe
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty. Keep finite.
    result = d + penalty
    # Ensure finiteness (defensive)
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    # Deadline guard: if we're running long, return a safe fallback
    if time.perf_counter() > _DEADLINE:
        return d.copy()

    return result
