"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a hybrid heuristic:
    - For small remaining sets (<=3), use exact lookahead
    - For larger sets, use a vectorized scoring heuristic balancing:
      * immediate distance from current node
      * distance to destination (for closing)
      * local connectivity penalty (min distance to other unvisited nodes)
    """
    # Deadline guard: if we're running out of time, fall back to nearest neighbor
    if time.time() - _start_time > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    unvisited = unvisited_nodes
    n_unvisited = len(unvisited)

    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited[0])

    # For 2 nodes: exact 2-step lookahead
    if n_unvisited == 2:
        i, j = unvisited
        cost_ij = distance_matrix[current_node, i] + distance_matrix[i, j] + distance_matrix[j, destination_node]
        cost_ji = distance_matrix[current_node, j] + distance_matrix[j, i] + distance_matrix[i, destination_node]
        return int(i if cost_ij <= cost_ji else j)

    # For 3 nodes: exact 3-step lookahead
    if n_unvisited == 3:
        best_node = unvisited[0]
        best_cost = float('inf')
        for first in unvisited:
            remaining = [x for x in unvisited if x != first]
            a, b = remaining[0], remaining[1]
            # Two possible orders for the remaining two
            cost1 = distance_matrix[current_node, first] + distance_matrix[first, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
            cost2 = distance_matrix[current_node, first] + distance_matrix[first, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
            cost = min(cost1, cost2)
            if cost < best_cost:
                best_cost = cost
                best_node = first
        return int(best_node)

    # For 4+ nodes: vectorized heuristic
    # Distances from current node to all unvisited nodes
    d_from_current = distance_matrix[current_node, unvisited]

    # Distances from all unvisited nodes to destination (for closing cost consideration)
    d_to_dest = distance_matrix[unvisited, destination_node]

    # Get the sub-matrix of distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited, unvisited)].copy()
    
    # For each row (candidate), the min distance to any other unvisited node
    # We need to exclude the diagonal (self-distance)
    np.fill_diagonal(sub_matrix, np.inf)
    min_next_from_candidate = np.min(sub_matrix, axis=1)

    # Adaptive weights: as fewer nodes remain, weight destination distance more heavily
    if n_unvisited <= 8:
        alpha = 0.4
        beta = 0.25
    elif n_unvisited <= 15:
        alpha = 0.25
        beta = 0.15
    elif n_unvisited <= 25:
        alpha = 0.15
        beta = 0.08
    else:
        alpha = 0.1
        beta = 0.04

    # Normalize components to comparable scales
    d_from_current_norm = d_from_current / (np.max(d_from_current) + 1e-9)
    d_to_dest_norm = d_to_dest / (np.max(d_to_dest) + 1e-9)
    min_next_norm = min_next_from_candidate / (np.max(min_next_from_candidate) + 1e-9)

    scores = d_from_current_norm + alpha * d_to_dest_norm + beta * min_next_norm

    best_idx = np.argmin(scores)
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
