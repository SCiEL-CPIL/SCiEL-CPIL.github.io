"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time once at module load
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 2.0  # return safe result this far before deadline

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: pick next customer or depot (0) to close route.
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node by:
         - immediate distance from current_node
         - plus a lookahead: minimum distance from that node to any other feasible node
           (encourages keeping the route connected/short overall)
         - plus a small depot-proximity term (helps if this is near the last stop)
      4. Pick the node with minimum combined score.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # If time is running out, fall back to simple nearest feasible
    if _time_exceeded():
        feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
        feasible_nodes = unvisited_nodes[feasible_mask]
        if len(feasible_nodes) == 0:
            return depot
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])

    # Filter by capacity
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Immediate distances from current node
    immediate_dists = distance_matrix[current_node, feasible_nodes]

    # Lookahead: min distance from each candidate to any other feasible node
    feas_idx = feasible_nodes
    sub_dist = distance_matrix[np.ix_(feas_idx, feas_idx)].copy()
    np.fill_diagonal(sub_dist, np.inf)
    min_other_dist = np.min(sub_dist, axis=1)

    # Distance to depot (for closing the route)
    depot_dist = distance_matrix[feasible_nodes, depot]

    # Combined score: immediate + alpha * min_other_dist + beta * depot_dist
    # We want to minimize total route length:
    #   - Short immediate distance is primary
    #   - Being close to other feasible nodes keeps future steps short
    #   - Being close to depot helps if this ends up being the last node
    alpha = 0.2
    beta = 0.1

    total_score = immediate_dists + alpha * min_other_dist + beta * depot_dist

    # Pick the node with minimum total score
    best_idx = np.argmin(total_score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
