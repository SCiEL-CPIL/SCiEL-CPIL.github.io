"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit to stop searching
_start_time = None


def _check_deadline():
    """Return True if we should stop searching (deadline approaching)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a connectivity-aware heuristic.
    Combines:
      1. Immediate distance from current node
      2. Distance from candidate to destination (final leg cost)
      3. Connectivity term: how well the candidate connects to remaining unvisited nodes
         (penalizes candidates that would create isolated clusters)
    """
    n = len(unvisited_nodes)
    if n == 0:
        return destination_node
    if n == 1:
        return int(unvisited_nodes[0])

    # If we're close to deadline, fall back to nearest neighbour for speed
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Base distances
    dists_current_to_cand = distance_matrix[current_node, unvisited_nodes]
    dists_cand_to_dest = distance_matrix[unvisited_nodes, destination_node]

    # Connectivity term: for each candidate, compute the minimum distance to any OTHER unvisited node
    # This penalizes candidates that are far from the rest of the unvisited set
    # (i.e., would create a disconnected or hard-to-complete subtour)
    if n > 2:
        # For each candidate i, find min distance to other unvisited nodes (excluding itself)
        # Use the distance matrix restricted to unvisited nodes
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(sub_matrix, np.inf)
        # Min distance from each candidate to any other unvisited node
        min_conn_dists = np.min(sub_matrix, axis=1)
    else:
        # With only 2 nodes, connectivity is just the distance between them
        min_conn_dists = np.array([distance_matrix[unvisited_nodes[0], unvisited_nodes[1]],
                                   distance_matrix[unvisited_nodes[1], unvisited_nodes[0]]])

    # Weighting:
    # - Early in the tour (large n): emphasize immediate distance and connectivity
    # - Late in the tour (small n): emphasize destination distance
    # alpha: weight on destination distance, increases as n decreases
    alpha = min(1.5, 3.0 / max(n, 1))
    # beta: weight on connectivity penalty, decreases as n decreases
    beta = max(0.1, 1.0 - n / 50.0)

    scores = dists_current_to_cand + alpha * dists_cand_to_dest + beta * min_conn_dists

    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
