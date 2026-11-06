"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(current_node)

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        # Exact 2-step lookahead for the final two cities
        n1, n2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        if cost1 <= cost2:
            return int(n1)
        else:
            return int(n2)

    # If time is running out, fall back to nearest neighbor for speed
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Structural change: 2-step lookahead combining nearest-neighbor extension
    # with a "gap-closing" term to destination.
    #
    # For each candidate i in unvisited_nodes:
    #   score(i) = dist(current, i) + min_{j in unvisited, j != i} [dist(i, j) + dist(j, destination)]
    #
    # This is a 2-step greedy: we extend the path by one city, then estimate
    # the best possible next step that also considers the return to destination.
    # This captures the structural idea from inspiration 2 (lookahead beyond 1 step)
    # with the refinement of explicitly including the destination gap (inspiration 1).

    dist_from_current = distance_matrix[current_node, unvisited_nodes]

    # dist_subset[i, j] = distance between unvisited_nodes[i] and unvisited_nodes[j]
    dist_subset = distance_matrix[unvisited_nodes][:, unvisited_nodes]

    # dist_to_dest[i] = distance between unvisited_nodes[i] and destination_node
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]

    # For each candidate i, compute: min over j != i of (dist_subset[i, j] + dist_to_dest[j])
    # This is the cost of going from i to some next city j, then to destination.
    # We want to add this to dist_from_current[i] as a lookahead penalty.

    # Build a matrix where entry [i, j] = dist_subset[i, j] + dist_to_dest[j]
    # Then for each row i, take the min over j (excluding j == i).
    lookahead_matrix = dist_subset + dist_to_dest[np.newaxis, :]

    # Set diagonal to infinity to exclude self
    np.fill_diagonal(lookahead_matrix, np.inf)

    # Min over axis=1 gives for each candidate i the best 2-step continuation
    best_lookahead = np.min(lookahead_matrix, axis=1)

    scores = dist_from_current + best_lookahead

    best_idx = np.argmin(scores)

    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
