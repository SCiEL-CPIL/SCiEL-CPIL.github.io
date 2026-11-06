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

    # If time is running out, fall back to nearest neighbor for speed
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Vectorized scoring: combine distance from current with "gap" to destination
    # The idea: we want a node that is close to current (to extend the path cheaply)
    # but also we want to avoid creating a very long jump to the destination later.
    # A good heuristic is to minimize: dist(current, candidate) + alpha * dist(candidate, destination)
    # where alpha balances local extension vs. future connection cost.
    
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Use a weighted sum. alpha=1.0 gives equal weight, which is a reasonable start.
    # This is essentially a "greedy with lookahead" or "2-choose" heuristic.
    scores = dist_from_current + dist_to_destination
    
    # Pick the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
