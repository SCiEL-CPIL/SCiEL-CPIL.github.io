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

    # Distance from each feasible node to the depot
    d_depot = distance_matrix[feas, depot]

    # For larger sets, use a smarter heuristic
    if k > 2:
        # Build a small (k x k) sub-matrix among feasible nodes
        sub = distance_matrix[np.ix_(feas, feas)]
        np.fill_diagonal(sub, np.inf)
        nearest_other = sub.min(axis=1)
        next_step_cost = np.minimum(nearest_other, d_depot)
        
        # Use a lower alpha to prioritize immediate distance more
        # This tends to produce tighter routes
        alpha = 0.3
        scores = d_cur + alpha * next_step_cost
    else:
        # For k == 2, just pick the one with lower immediate + return cost
        # This avoids the overhead of building a sub-matrix
        # Candidate 0: go to feas[0], then min(distance to feas[1], distance to depot)
        # Candidate 1: go to feas[1], then min(distance to feas[0], distance to depot)
        d01 = distance_matrix[feas[0], feas[1]]
        d10 = d01  # symmetric
        
        # After visiting feas[0], next is min(d01, d_depot[0])
        # After visiting feas[1], next is min(d10, d_depot[1])
        next0 = min(d01, d_depot[0])
        next1 = min(d10, d_depot[1])
        
        score0 = d_cur[0] + 0.5 * next0
        score1 = d_cur[1] + 0.5 * next1
        
        if score0 <= score1:
            return int(feas[0])
        else:
            return int(feas[1])

    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
