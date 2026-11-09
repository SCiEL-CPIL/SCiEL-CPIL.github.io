"""CVRP heuristic: savings-based next-node selection with look-ahead and capacity awareness."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = None
_DEADLINE = 112.0  # safety margin before the 120s wall-clock kill


def _now():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: stop exploring and just return something safe.
    if _now() > _DEADLINE:
        # If we can't compute, return depot to close the route (always feasible).
        return 0

    n = int(len(unvisited_nodes))
    if n == 0:
        return 0

    # Convert unvisited to a numpy array of ints for fast indexing.
    unv = np.asarray(unvisited_nodes, dtype=np.int64)

    # Feasibility: demand of candidate must fit remaining capacity.
    cand_demand = demands[unv]
    feasible_mask = cand_demand <= rest_capacity
    feasible_count = int(np.count_nonzero(feasible_mask))

    if feasible_count == 0:
        return 0

    # If only one feasible, take it.
    if feasible_count == 1:
        return int(unv[feasible_mask][0])

    # Distances from current node to all unvisited.
    d_from_cur = distance_matrix[current_node, unv]
    # Distances from each unvisited back to depot (cost to close after visiting).
    d_to_depot = distance_matrix[unv, depot]

    # Score each feasible candidate.
    # We want to prefer nodes that keep the route "open" cheaply.
    # Savings idea: continuing to j costs d(cur,j) + (future). Closing now costs d(cur,depot).
    # A good proxy: prefer small d(cur,j) but penalize nodes that are far from depot
    # (because visiting them forces a long return), and prefer nodes that are close
    # to other remaining feasible nodes (to chain efficiently).

    # Base: distance from current to candidate.
    base = d_from_cur.copy()

    # Penalty for being far from depot (forces long route closure).
    base = base + 0.3 * d_to_depot

    # Look-ahead: for each candidate j, estimate the cheapest next step to another
    # feasible node k (or depot). Prefer candidates that have a cheap continuation.
    # This is O(n^2) but n<=50, so fine.
    # We compute min over other feasible nodes of d(j,k) + 0.3*d(k,depot),
    # and also consider going to depot: d(j,depot).
    # We add a fraction of this continuation cost to the score.
    feas_idx = np.where(feasible_mask)[0]
    # Matrix of distances among unvisited nodes (n x n).
    d_unv = distance_matrix[np.ix_(unv, unv)]

    cont = np.full(n, np.inf)
    for fi in feas_idx:
        # continuation to depot
        c_depot = d_to_depot[fi]
        # continuation to other feasible nodes
        if feasible_count > 1:
            others = feas_idx[feas_idx != fi]
            c_other = d_unv[fi, others] + 0.3 * d_to_depot[others]
            c_best = min(c_depot, float(np.min(c_other)))
        else:
            c_best = c_depot
        cont[fi] = c_best

    score = base + 0.4 * cont

    # Restrict to feasible.
    score[~feasible_mask] = np.inf

    best_idx = int(np.argmin(score))
    return int(unv[best_idx])
# EVOLVE-BLOCK-END
