"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_START = time.time()
_TIME_LIMIT = 110.0  # safety margin under the 120 s wall-clock kill


def _elapsed():
    return time.time() - _DEADLINE_START


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: if we are about to exceed the budget, close the route.
    if _elapsed() > _TIME_LIMIT:
        return int(depot)

    unvisited_nodes = np.asarray(unvisited_nodes)
    if unvisited_nodes.size == 0:
        return int(depot)

    dmat = np.asarray(distance_matrix)
    cur = int(current_node)
    dep = int(depot)

    # Demand feasibility: candidate must fit in remaining capacity.
    dem = np.asarray(demands)
    feasible_mask = dem[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return int(dep)

    feasible = unvisited_nodes[feasible_mask]

    # Distances from current to feasible candidates, and current->depot.
    d_cur = dmat[cur, feasible]
    d_cur_dep = dmat[cur, dep]

    # Distances from each feasible candidate back to depot (surrogate for
    # eventual return cost; smaller means the customer is "closer to home").
    d_cand_dep = dmat[feasible, dep]

    # Savings-style surrogate: prefer candidates that are close to current AND
    # close to depot (central, short return leg). Weights balance the two.
    # Normalize d_cand_dep against d_cur scale so the two terms are comparable.
    scale = max(float(np.max(d_cur)), 1e-9)
    score = 1.0 * d_cur + 0.5 * (d_cand_dep / scale)

    idx = int(np.argmin(score))
    best_node = int(feasible[idx])
    best_score = float(score[idx])

    # Decide whether to close the route instead: if the best customer is
    # "worse" than going straight back to depot (accounting for the return
    # leg we would still pay), return to depot.
    # Cost of going to best then (eventually) depot vs. going depot now:
    # marginal = d_cur[idx] + d_cand_dep[idx]  (visit + return)
    # vs. close = d_cur_dep (return now)
    # If visiting is clearly not cheaper than closing, close the route.
    if (float(d_cur[idx]) + float(d_cand_dep[idx])) > float(d_cur_dep):
        return int(dep)

    return best_node
# EVOLVE-BLOCK-END
