"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node using a combined metric:
         - immediate distance from current_node
         - savings penalty (extra cost vs going straight to depot)
         - soft-min lookahead: smooth surrogate for min distance to any other feasible node or depot
         - capacity utilization bonus: encourage fuller trucks
      This balances proximity, route efficiency, avoids dead-end picks, and promotes capacity usage.
    """
    if _time_exceeded():
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    feasible_arr = np.array(feasible, dtype=np.int64)
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Immediate distances from current_node
    dist_from_current = distance_matrix[current_node, feasible_arr]

    # Savings penalty: dist(current->node) + dist(node->depot) - dist(current->depot)
    dist_to_depot = distance_matrix[feasible_arr, depot]
    dist_current_to_depot = distance_matrix[current_node, depot]
    savings_penalty = dist_from_current + dist_to_depot - dist_current_to_depot

    # Vectorized soft-min lookahead: for each feasible node, compute a smooth
    # surrogate for the minimum distance to any other feasible node or depot.
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]

    t = 2.5  # temperature for soft-min
    diag_mask = np.eye(n_feasible + 1, dtype=bool)
    masked_sub = sub_dist.copy()
    masked_sub[diag_mask] = 1e9

    # Numerical stability: subtract row-wise min
    row_min = np.min(masked_sub, axis=1)
    shifted = masked_sub - row_min[:, np.newaxis]
    exp_vals = np.exp(-shifted / t)
    exp_vals[diag_mask] = 0.0
    s = np.sum(exp_vals, axis=1)
    lookahead = row_min[:n_feasible] + t * np.log(s[:n_feasible])

    # Capacity utilization bonus: prefer nodes that use more of the remaining capacity
    # This encourages fuller trucks, reducing total number of routes
    cap_bonus = -0.055 * (demands[feasible_arr] / rest_capacity)

    # Combined score
    alpha_savings = 0.27
    alpha_lookahead = 0.38
    scores = dist_from_current + alpha_savings * savings_penalty + alpha_lookahead * lookahead + cap_bonus

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
