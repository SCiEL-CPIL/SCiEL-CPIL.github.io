"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_EVAL_START = time.time()
DEADLINE = _EVAL_START + 140  # Safety margin before 150s kill


def _time_remaining():
    return DEADLINE - time.time()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a refined savings-inspired heuristic with
    adaptive depot-proximity weighting based on remaining capacity.
    - Filter to feasible customers (demand <= rest_capacity)
    - Score each candidate with adaptive weights:
      * Strong preference for proximity to current node
      * Adaptive depot-proximity bonus (stronger when capacity is tight)
      * Mild demand-preservation penalty when capacity is very tight
    - Return depot if no feasible customer remains
    - Deadline guard to ensure we return before timeout
    """
    # Deadline guard
    if _time_remaining() < 1.0:
        # Emergency fallback: nearest feasible node
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # If no unvisited nodes, return depot
    if len(unvisited_nodes) == 0:
        return 0

    # Filter to only feasible customers (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible = unvisited_arr[feasible_mask]

    # If no feasible customer, return depot to close route
    if len(feasible) == 0:
        return 0

    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute distances
    dist_to_node = distance_matrix[current_node, feasible]
    dist_to_depot = distance_matrix[feasible, depot]

    # Adaptive depot weight:
    # When rest_capacity is small relative to typical demand, we should
    # more strongly prefer nodes close to the depot to ensure we can
    # close the route without wasting capacity on distant nodes.
    # Use a smooth transition: as rest_capacity decreases, depot_weight increases.
    # With capacity 50 and demands 1-9, rest_capacity ranges from ~1 to 50.
    # Map rest_capacity to a weight in [0.3, 1.2]:
    #   weight = 0.3 + 0.9 * (1 - rest_capacity / 50) clipped
    cap_ratio = rest_capacity / 50.0
    depot_weight = 0.3 + 0.9 * (1.0 - cap_ratio)
    # Clip to reasonable range
    depot_weight = max(0.2, min(1.5, depot_weight))

    # Base score: strong proximity to current, adaptive depot proximity
    # score = dist[current][i] + depot_weight * dist[depot][i]
    # Lower is better.
    scores = dist_to_node + depot_weight * dist_to_depot

    # Demand-preservation penalty:
    # When capacity is very tight (rest_capacity < 15), add a small penalty
    # for high-demand nodes to preserve flexibility for later customers.
    # This is a soft penalty: penalty = 0.02 * demand[i] * (1 - rest_capacity/15)
    # Only active when rest_capacity < 15.
    if rest_capacity < 15:
        demand_penalty_factor = 0.02 * (1.0 - rest_capacity / 15.0)
        if demand_penalty_factor > 0:
            feasible_demands = demands[feasible]
            scores = scores + demand_penalty_factor * feasible_demands

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
