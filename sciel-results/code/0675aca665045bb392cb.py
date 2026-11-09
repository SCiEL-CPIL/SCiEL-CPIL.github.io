"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 110.0  # safety margin before 120s kill
_start_time = None


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.monotonic() - _start_time


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.monotonic()

    # Deadline guard: if we're running out of time, just pick nearest feasible
    if _elapsed() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        return int(feible[np.argmin(distance_matrix[current_node, feasible])])

    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return int(depot)

    # Composite scoring: balance distance to node, continuation cost (min dist from
    # that node to other feasible nodes or depot), and demand penalty.
    # This encourages tight clustered routes.
    
    n = len(feasible)
    if n == 1:
        return int(feasible[0])

    # Convert to numpy for vectorized ops
    feas_arr = np.array(feasible)
    dists_from_current = distance_matrix[current_node, feas_arr]
    demands_arr = demands[feas_arr]

    # Compute continuation cost: for each candidate, the minimum distance from that
    # candidate to any other feasible node or the depot
    # This penalizes picking a node that leaves you far from remaining feasible customers
    min_cont_costs = np.zeros(n)
    for i in range(n):
        node = feas_arr[i]
        # min distance from this node to other feasible nodes (excluding itself) and depot
        if n > 1:
            other_dists = distance_matrix[node, feas_arr]
            other_dists[i] = np.inf  # exclude self
            min_to_others = np.min(other_dists)
        else:
            min_to_others = 0.0
        dist_to_depot = distance_matrix[node, depot]
        min_cont_costs[i] = min(min_to_others, dist_to_depot)

    # Score: lower is better
    # Weighted combination: primary is distance from current, secondary is continuation cost
    # Tertiary: prefer higher demand nodes early (fill capacity efficiently)
    score = (1.0 * dists_from_current + 
             0.6 * min_cont_costs + 
             0.05 * (10.0 - demands_arr))  # small penalty for low-demand nodes

    best_idx = np.argmin(score)
    return int(feas_arr[best_idx])
# EVOLVE-BLOCK-END
