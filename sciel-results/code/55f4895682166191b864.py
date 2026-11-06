"""Spatial-clustering heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin under 150s wall-clock
_START = time.time()
_SOFTMIN_T = 4.0   # temperature for soft-min/soft-max surrogate


def _elapsed():
    return time.time() - _START


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: if we're close to the limit, fall back to nearest feasible.
    if _elapsed() > _DEADLINE:
        feas = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feas:
            return 0
        return int(min(feas, key=lambda n: distance_matrix[current_node, n]))

    unvisited = np.asarray(unvisited_nodes)
    if unvisited.size == 0:
        return 0

    # Feasible candidates: demand fits in remaining capacity.
    feas_mask = demands[unvisited] <= rest_capacity
    feas = unvisited[feas_mask]
    if feas.size == 0:
        return 0

    d = np.asarray(distance_matrix)
    dm_feas = d[current_node, feas]          # distance current -> candidate

    # --- Cluster density via log-sum-exp (soft surrogate) ---
    # For each candidate i, soft-sum of closeness to all other feasible nodes j.
    # closeness = 1 / (1 + dist(i,j)); log-sum-exp gives a smooth "density" measure.
    if feas.size > 1:
        d_ij = d[np.ix_(feas, feas)]
        np.fill_diagonal(d_ij, 0.0)
        closeness = 1.0 / (1.0 + d_ij)
        # log-sum-exp over axis 1 (per candidate)
        max_c = closeness.max(axis=1, keepdims=True)
        log_sum_exp = max_c[:, 0] + np.log(np.sum(np.exp(closeness - max_c), axis=1))
        density = log_sum_exp
    else:
        density = np.zeros(1)

    # Normalize density to [0,1]
    d_min, d_max = density.min(), density.max()
    if d_max - d_min > 1e-12:
        density_norm = (density - d_min) / (d_max - d_min)
    else:
        density_norm = np.ones_like(density)

    # --- Edge savings (Clarke-Wright style local) ---
    # savings[i] = dist(current, feas[i]) + dist(feas[i], best_next) - dist(current, best_next)
    # Approximate: for each candidate, find its nearest other feasible node.
    if feas.size > 1:
        d_ij = d[np.ix_(feas, feas)]
        np.fill_diagonal(d_ij, np.inf)
        nearest_other = d_ij.min(axis=1)
        # best next from current (overall nearest feasible)
        nearest_from_cur = dm_feas.min()
        # savings: going current->i->nearest(i) vs current->nearest(i)
        savings = dm_feas + nearest_other - nearest_from_cur
        # Higher savings = better (we save distance by inserting i before its neighbor)
        s_min, s_max = savings.min(), savings.max()
        if s_max - s_min > 1e-12:
            savings_norm = (savings - s_min) / (s_max - s_min)
        else:
            savings_norm = np.ones_like(savings)
    else:
        savings_norm = np.ones(1)

    # --- Capacity consolidation term ---
    # Reward picking a node that lets us use capacity efficiently in the cluster.
    # Smooth penalty: if after picking i, remaining capacity can't cover nearby demand, penalize.
    # Estimate: sum of demands of feasible nodes within a radius of candidate i.
    if feas.size > 1:
        d_ij = d[np.ix_(feas, feas)]
        # Soft count of nearby demand: weight by exp(-dist / scale)
        scale = max(1.0, dm_feas.mean() * 0.5)
        weight = np.exp(-d_ij / scale)
        np.fill_diagonal(weight, 1.0)
        local_demand = weight @ demands[feas]
        # After picking i, remaining capacity = rest_capacity - demands[i]
        rem_cap_after = rest_capacity - demands[feas]
        # Fraction of local demand we can still cover (capped at 1)
        coverage = np.minimum(local_demand / np.maximum(rem_cap_after, 1e-9), 1.0)
        # We want high coverage (can serve most of the local cluster)
        cap_score = coverage
    else:
        cap_score = np.ones(1)

    # --- Distance penalty (prefer closer nodes) ---
    dist_min, dist_max = dm_feas.min(), dm_feas.max()
    if dist_max - dist_min > 1e-12:
        dist_norm = (dm_feas - dist_min) / (dist_max - dist_min)
    else:
        dist_norm = np.zeros_like(dm_feas)

    # --- Combined score ---
    # Lower is better. We combine:
    #   - distance (lower better)
    #   - density (higher better -> subtract)
    #   - savings (higher better -> subtract)
    #   - capacity coverage (higher better -> subtract)
    w_dist = 0.35
    w_density = 0.25
    w_savings = 0.25
    w_cap = 0.15

    score = (w_dist * dist_norm
             - w_density * density_norm
             - w_savings * savings_norm
             - w_cap * cap_score)

    # Smooth min via log-sum-exp for robustness (pick argmin of score)
    best_idx = int(np.argmin(score))

    return int(feas[best_idx])
# EVOLVE-BLOCK-END
