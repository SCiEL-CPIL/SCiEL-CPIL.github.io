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

    Scores each feasible customer by a weighted combination of:
      - distance from current node to candidate
      - distance from candidate back to depot (return cost)
      - demand (prefer lower demand to preserve capacity)
    Returns the best customer, or the depot (0) to close the route.
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

    # Composite scoring.
    #   term1: distance current -> candidate
    #   term2: distance candidate -> depot (return cost, weighted)
    #   term3: demand penalty (prefer low demand to keep capacity for more visits)
    d_cur = distance_matrix[current_node, feasible]
    d_dep = distance_matrix[feasible, depot]
    d_dem = demands[feasible].astype(float)

    # Normalize-ish weights. Distances are O(1..~30) for 50-customer instances;
    # demands 1..9. We want distance to dominate but return-cost and demand to
    # matter.
    w_return = 0.5
    w_demand = 0.0

    # Scale demand penalty by 1/rest_capacity so it's a fraction of remaining cap.
    if rest_capacity > 0:
        demand_frac = d_dem / rest_capacity
    else:
        demand_frac = np.zeros_like(d_dem)

    score = d_cur + w_return * d_dep + w_demand * demand_frac

    # Choose the minimum-score candidate.
    best_idx = int(np.argmin(score))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
