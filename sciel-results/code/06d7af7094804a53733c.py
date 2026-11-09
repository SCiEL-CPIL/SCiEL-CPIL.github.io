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
    Select the next node using a 2-step lookahead heuristic.
    Score = dist(current, candidate) + alpha * dist(candidate, destination)
    This avoids myopic choices that lead to expensive final connections.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return destination_node
    if n == 1:
        return int(unvisited_nodes[0])

    # If we're close to deadline, fall back to nearest neighbour for speed
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # 2-step lookahead: score each candidate by
    # dist(current, candidate) + dist(candidate, destination)
    # This balances immediate cost with the cost of connecting to the final destination.
    dists_current_to_cand = distance_matrix[current_node, unvisited_nodes]
    dists_cand_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Use a weighted sum; weight on destination distance decreases as more nodes remain
    # Early in the tour, destination is far away so we weight it less
    # Late in the tour, destination matters more
    alpha = min(1.0, 2.0 / max(n, 1))
    scores = dists_current_to_cand + alpha * dists_cand_to_dest
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
