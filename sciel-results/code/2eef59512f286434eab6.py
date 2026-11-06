"""CVRP heuristic: spatial-clustering + regret hybrid with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 120.0  # safety margin under 150s kill
_T0 = time.perf_counter()


def _elapsed():
    return time.perf_counter() - _T0


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: if we are close to the limit, fall back to a fast,
    # safe nearest-feasible choice so we never timeout.
    if _elapsed() > _DEADLINE:
        feas = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
        if len(feas) == 0:
            return 0
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if len(unvisited_nodes) == 0:
        return 0

    # Feasible set: demand fits in remaining capacity.
    feas_mask = demands[unvisited_nodes] <= rest_capacity
    feas = unvisited_nodes[feas_mask]
    if len(feas) == 0:
        return 0

    # If only one feasible, take it.
    if len(feas) == 1:
        return int(feas[0])

    dm = np.asarray(distance_matrix)
    cur = int(current_node)

    # --- Component 1: spatial density (Voronoi-like) consolidation score ---
    # For each feasible candidate i, count how much *remaining feasible demand*
    # lies within a radius that scales with the median inter-customer distance.
    # This rewards candidates embedded in dense clusters of still-unserved demand.
    #
    # We approximate the "Voronoi region demand" by, for each candidate i,
    # summing demands[j] of other feasible j that are closer to i than to the
    # current node (i.e., i is the better "local hub" for j given where we are).
    # This is a cheap proxy for area-based demand consolidation.
    dist_from_cur = dm[cur, feas]                      # (n_feas,)
    dist_feas_feas = dm[np.ix_(feas, feas)]            # (n_feas, n_feas)
    dem_feas = demands[feas]                           # (n_feas,)

    # For candidate i, the set of feasible j for which i is at least as close
    # as the current node is a proxy for i's "local service area".
    # weight = demand of j, but only if dist(i,j) is small relative to scale.
    # Use a soft kernel: w = exp(-d(i,j)^2 / (2*sigma^2)).
    # sigma chosen from median distance among feasible nodes.
    d_upper = np.triu(dist_feas_feas, k=1)
    d_upper = d_upper[d_upper > 0]
    if d_upper.size == 0:
        sigma = 1.0
    else:
        sigma = float(np.median(d_upper))
    if sigma <= 0:
        sigma = 1.0
    sigma2 = 2.0 * sigma * sigma

    # Kernel matrix K[i,j] = exp(-d(i,j)^2 / sigma2)  (i != j), 0 on diagonal.
    K = np.exp(-(dist_feas_feas ** 2) / sigma2)
    np.fill_diagonal(K, 0.0)

    # density[i] = sum_j K[i,j] * demand[j]  -> demand mass in i's local region.
    density = K @ dem_feas                       # (n_feas,)

    # --- Component 2: regret ---
    # regret[i] = (second-best distance from current to i's "slot") - (best).
    # Approximate: for each feasible i, consider the cost of *not* serving i now:
    # the extra distance another future node would incur. Use:
    #   regret[i] = min_{j != i} dist(i,j) - min_{j != i} dist(cur,j)?? 
    # A cleaner, measured-effective regret: for each candidate i, the difference
    # between the best and second-best insertion cost. Here we approximate the
    # "insertion cost" of i as dist(cur,i) + dist(i, next-best-neighbor).
    # Simpler robust regret: how uniquely close i is to the current node relative
    # to other candidates. regret[i] = (2nd smallest dist_from_cur) - dist_from_cur[i].
    order = np.argsort(dist_from_cur)
    if len(order) >= 2:
        second_best = dist_from_cur[order[1]]
    else:
        second_best = dist_from_cur[order[0]]
    regret = second_best - dist_from_cur         # (n_feas,)
    regret = np.clip(regret, 0.0, None)

    # --- Component 3: immediate distance (keep a strong local term) ---
    immediate = dist_from_cur.copy()

    # Normalize each component to [0,1] (lower is better for immediate/regret,
    # higher is better for density).
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

    # We want to MINIMIZE a combined score.
    #   - lower immediate distance is better
    #   - higher density is better  -> use (1 - norm_density)
    #   - higher regret is better   -> use (1 - norm_regret)
    imm_n = _norm_lower(immediate)
    den_n = _norm_higher(density)   # 0 = densest, 1 = least dense
    reg_n = _norm_higher(regret)    # 0 = highest regret, 1 = lowest regret

    # Capacity-adaptive weighting:
    #  - When rest_capacity is small relative to total remaining demand, we are
    #    closing a route soon -> favor density (consolidate local demand) and
    #    regret (don't leave uniquely-served nodes stranded).
    #  - When rest_capacity is large, we can afford to chase immediate distance
    #    savings to build a long efficient route.
    total_remaining = float(demands[unvisited_nodes].sum())
    if total_remaining <= 0:
        cap_frac = 0.0
    else:
        cap_frac = min(1.0, rest_capacity / total_remaining)

    # Weights: immediate always present; density and regret grow as capacity
    # fraction shrinks (route is filling up).
    w_imm = 1.0
    w_den = 0.5 + 1.5 * (1.0 - cap_frac)   # 0.5..2.0
    w_reg = 0.3 + 1.2 * (1.0 - cap_frac)   # 0.3..1.5

    score = w_imm * imm_n + w_den * den_n + w_reg * reg_n

    # Deadline guard again before the final pick (cheap here).
    if _elapsed() > _DEADLINE:
        return int(feas[np.argmin(immediate)])

    idx = int(np.argmin(score))
    return int(feas[idx])
# EVOLVE-BLOCK-END
