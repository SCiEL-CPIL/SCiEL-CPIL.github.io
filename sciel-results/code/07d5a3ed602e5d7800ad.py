"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    # Feasible: demand fits in remaining capacity
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    k = len(feas)

    # Distance from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]

    if k == 1:
        return int(feas[0])

    # Clarke-Wright savings heuristic:
    # savings[i] = d(current, depot) + d(depot, feas[i]) - d(current, feas[i])
    # Higher savings means visiting this node saves more distance compared to
    # going back to depot and then visiting it separately.
    d_cur_depot = distance_matrix[current_node, depot]
    d_depot_feas = distance_matrix[depot, feas]
    savings = d_cur_depot + d_depot_feas - d_cur

    # Also consider capacity: prefer nodes that leave more room for future visits
    # Normalize demands to [0,1] range for blending
    d_feas = demands[feas]
    max_d = d_feas.max()
    if max_d > 0:
        norm_d = d_feas / max_d
    else:
        norm_d = np.zeros_like(d_feas, dtype=float)

    # Lookahead: for each candidate, estimate the cost of the next leg by
    # finding the nearest other feasible node (excluding the candidate itself).
    # This penalizes choices that leave you stranded far from remaining customers.
    if k > 1:
        # Distance from each feasible node to all other feasible nodes
        feas_dm = distance_matrix[np.ix_(feas, feas)]
        # Mask out self-distances
        np.fill_diagonal(feas_dm, np.inf)
        # Nearest neighbor distance for each feasible node
        nn_dist = feas_dm.min(axis=1)
    else:
        nn_dist = np.zeros(1)

    # Dynamic regret-minimization: penalize outlier-long edges relative to
    # the average edge cost of the remaining feasible set. This steers the
    # route away from unusually long edges to maintain uniform segment lengths
    # and prevent premature capacity exhaustion on distant nodes.
    mean_d_cur = np.mean(d_cur)
    regret = np.maximum(0, d_cur - mean_d_cur)

    # Combine: minimize (d_cur - beta * savings + gamma * norm_d * d_cur_depot + delta * nn_dist + epsilon * regret)
    # The savings term rewards compactness, the demand term slightly penalizes
    # high-demand nodes when capacity is tight, the lookahead term penalizes
    # nodes that are isolated from other remaining customers, and the regret
    # term penalizes outlier-long edges.
    beta = 0.8
    gamma = 0.15
    delta = 0.3
    epsilon = 0.2
    scores = d_cur - beta * savings + gamma * norm_d * d_cur_depot + delta * nn_dist + epsilon * regret

    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
