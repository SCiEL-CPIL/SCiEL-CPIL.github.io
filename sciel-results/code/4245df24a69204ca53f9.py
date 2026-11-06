"""CVRP heuristic: route-projection + angular-sector density with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 120.0
_T0 = time.perf_counter()


def _elapsed():
    return time.perf_counter() - _T0


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: fast nearest-feasible fallback.
    if _elapsed() > _DEADLINE:
        feas = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
        if len(feas) == 0:
            return 0
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if len(unvisited_nodes) == 0:
        return 0

    feas_mask = demands[unvisited_nodes] <= rest_capacity
    feas = unvisited_nodes[feas_mask]
    if len(feas) == 0:
        return 0
    if len(feas) == 1:
        return int(feas[0])

    dm = np.asarray(distance_matrix)
    cur = int(current_node)
    dep = int(depot)

    dist_from_cur = dm[cur, feas]
    dist_to_depot = dm[feas, dep]
    dem_feas = demands[feas].astype(float)

    # --- Route-projection term: cost if we go to i then must return to depot ---
    # A candidate that makes the route infeasible to close (dist_to_depot too
    # large relative to remaining budget after serving i) is penalised.
    # Use a soft penalty: proj[i] = dist_from_cur[i] + max(0, dist_to_depot[i] - slack)
    # where slack is a fraction of rest_capacity (proxy for how far we can still travel).
    # Since distances are arbitrary units, normalise by median distance.
    all_d = dm[np.ix_(feas, feas)]
    d_upper = np.triu(all_d, k=1)
    d_upper = d_upper[d_upper > 0]
    if d_upper.size == 0:
        med_d = 1.0
    else:
        med_d = float(np.median(d_upper))
    if med_d <= 0:
        med_d = 1.0

    # Slack: how much "distance budget" remains after serving i.
    # We approximate that serving i uses demand[i] worth of capacity, and the
    # distance budget scales linearly with capacity (a heuristic).
    # slack_i = (rest_capacity - demand[i]) / rest_capacity * (2 * med_d)
    # If dist_to_depot[i] > slack_i, the route may not close -> penalty.
    cap_ratio = (rest_capacity - dem_feas) / max(rest_capacity, 1e-9)
    slack = cap_ratio * (2.0 * med_d)
    penalty = np.maximum(0.0, dist_to_depot - slack)
    proj = dist_from_cur + penalty

    # --- Angular-sector density: demand mass in a cone ahead of the current node ---
    # For each candidate i, define a forward cone from cur through i with
    # half-angle 60 degrees. Sum demands of other feasible nodes j that lie
    # inside this cone. This rewards extending the route into a dense direction.
    # Use dot-product angle test (no sqrt needed beyond normalisation).
    # Represent nodes by their distance-matrix coordinates? We don't have coords.
    # Instead, use a distance-based proxy: j is "in the sector" of i if
    # dist(cur, j) < dist(cur, i) + 0.5 * dist(i, j)  (triangle-inequality sector).
    # This is a cheap geometric proxy without explicit coordinates.
    n = len(feas)
    dist_cur_all = dist_from_cur  # (n,)
    d_ii = all_d  # (n, n)
    # sector_mask[i, j] = (dist_cur_all[j] < dist_cur_all[i] + 0.5 * d_ii[i, j])
    # and j != i
    sector_mask = (dist_cur_all[None, :] < dist_cur_all[:, None] + 0.5 * d_ii)
    np.fill_diagonal(sector_mask, False)
    sector_density = sector_mask @ dem_feas  # (n,)

    # --- Normalise ---
    def _norm_lower(x):
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _norm_higher(x):
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (hi - x) / (hi - lo)

    proj_n = _norm_lower(proj)
    den_n = _norm_higher(sector_density)

    # Capacity-adaptive weighting:
    total_remaining = float(demands[unvisited_nodes].sum())
    if total_remaining <= 0:
        cap_frac = 0.0
    else:
        cap_frac = min(1.0, rest_capacity / total_remaining)

    # When capacity is nearly full, penalise route-projection heavily (must close soon).
    # When capacity is abundant, favour sector density to build tight clusters.
    w_proj = 1.0 + 2.0 * (1.0 - cap_frac)
    w_den = 0.5 + 1.0 * cap_frac

    score = w_proj * proj_n + w_den * den_n

    if _elapsed() > _DEADLINE:
        return int(feas[np.argmin(dist_from_cur)])

    idx = int(np.argmin(score))
    return int(feas[idx])
# EVOLVE-BLOCK-END
