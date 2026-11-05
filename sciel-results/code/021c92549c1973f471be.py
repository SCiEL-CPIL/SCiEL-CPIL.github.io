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
    Select the next node using a combined heuristic:
    - Filter to feasible customers (demand <= rest_capacity)
    - Score each candidate by a weighted combination of distance to current
      and distance to depot (savings-inspired insertion cost)
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

    # Score each feasible candidate:
    # score = dist[current][node] + lambda * dist[node][depot]
    # This approximates the incremental cost of visiting this node,
    # slightly favoring nodes that are also close to the depot (shorter routes).
    # The savings heuristic insight: a node that is close to both current
    # and depot represents a "savings" opportunity.
    dist_to_node = distance_matrix[current_node, feasible]
    dist_to_depot = distance_matrix[feasible, depot]

    # Weight: 0.3 favors nodes close to depot (savings-inspired)
    scores = dist_to_node + 0.3 * dist_to_depot

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
