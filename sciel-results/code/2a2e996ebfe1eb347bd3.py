"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track start time and bail out to a safe (nearest-feasible or depot) choice
# if we approach the limit, leaving a safety margin.
_START_TIME = None
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0  # seconds


def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _get_start_time())


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for Capacitated VRP.

    Uses a 2-step lookahead with Clarke-Wright savings and capacity-aware scoring.
    For each feasible candidate, we estimate the cost of visiting it now plus the
    best subsequent move (to another feasible node or back to depot). This captures
    local tightness and route continuity better than single-step savings.
    """
    # Deadline guard: if we're about to run out of time, fall back to a
    # fast, safe choice.
    if _time_remaining() < 0.5:
        # Fast fallback: nearest feasible customer, else depot.
        if len(unvisited_nodes) == 0:
            return 0
        feasible = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return 0

    # Filter to feasible customers (demand fits remaining capacity).
    feasible = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible) == 0:
        return 0

    n_feas = len(feasible)
    if n_feas == 1:
        return int(feasible[0])

    # Convert to numpy arrays for vectorized operations
    d_cur = distance_matrix[current_node, feasible]
    d_dep = distance_matrix[feasible, depot]
    d_cur_dep = distance_matrix[current_node, depot]
    d_dem = demands[feasible].astype(float)

    # 2-step lookahead: for each candidate i, compute the minimum cost to go
    # from i to some other feasible node j (with remaining capacity after i)
    # or back to depot. This is a proxy for "how well does this choice set up
    # the next step?"

    # For each candidate i, after visiting it, remaining capacity is rest_capacity - d_dem[i]
    # The next node j must satisfy d_dem[j] <= rest_capacity - d_dem[i]
    # We approximate this by computing, for each i, the min distance from i to:
    #   - any feasible j where d_dem[j] <= rest_capacity - d_dem[i]
    #   - depot (if we close the route after i)
    # Then the 2-step cost is d_cur[i] + min_next[i]

    # Build a matrix of distances from each feasible node to all other feasible nodes
    # dist_feas[i, j] = distance_matrix[feasible[i], feasible[j]]
    dist_feas = distance_matrix[np.ix_(feasible, feasible)]

    # For each i, the set of valid next nodes j are those where d_dem[j] <= rest_capacity - d_dem[i]
    # We want min over valid j of dist_feas[i, j], and also consider returning to depot (d_dep[i])
    # So min_next[i] = min( min_{j: d_dem[j] <= rest_cap - d_dem[i]} dist_feas[i,j], d_dep[i] )

    # Vectorized approach:
    # For each i, create a mask of valid j's
    # remaining_after_i = rest_capacity - d_dem[i]  (shape: n_feas,)
    # valid_mask[i, j] = d_dem[j] <= remaining_after_i[i]
    remaining_after_i = rest_capacity - d_dem  # shape: (n_feas,)
    # d_dem is shape (n_feas,), remaining_after_i is shape (n_feas,)
    # valid_mask[i, j] = d_dem[j] <= remaining_after_i[i]
    # This is: d_dem[None, :] <= remaining_after_i[:, None]
    valid_mask = d_dem[None, :] <= remaining_after_i[:, None]  # shape: (n_feas, n_feas)

    # For each row i, find min dist_feas[i, j] over valid j's
    # Set invalid entries to inf, then take min along axis 1
    dist_feas_masked = np.where(valid_mask, dist_feas, np.inf)
    min_next_to_feasible = np.min(dist_feas_masked, axis=1)  # shape: (n_feas,)

    # Also consider going back to depot after i
    min_next = np.minimum(min_next_to_feasible, d_dep)

    # 2-step cost for candidate i: d_cur[i] + min_next[i]
    two_step_cost = d_cur + min_next

    # Combine with savings-inspired scoring:
    # savings = d_cur_dep + d_dep[i] - d_cur[i]
    # We want to maximize savings, so minimize -savings = d_cur[i] - d_dep[i] - d_cur_dep
    # Since d_cur_dep is constant, minimize d_cur[i] - d_dep[i]
    # But we also want to consider the 2-step cost and demand.

    # Composite score:
    #   score = alpha * two_step_cost + beta * (d_cur - d_dep) + gamma * demand_frac
    # Where:
    #   two_step_cost captures local continuity
    #   (d_cur - d_dep) captures savings (negative means good savings)
    #   demand_frac penalizes large demands to preserve capacity

    alpha = 1.0
    beta = 0.5
    gamma = 0.1

    if rest_capacity > 0:
        demand_frac = d_dem / rest_capacity
    else:
        demand_frac = np.zeros_like(d_dem)

    score = alpha * two_step_cost + beta * (d_cur - d_dep) + gamma * demand_frac

    # Pick the candidate with minimum score
    best_idx = int(np.argmin(score))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
