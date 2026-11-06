"""Traveling salesman constructive heuristic with destination look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 110.0  # Safety margin from 120s limit


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
    """
    Select the next node using a look-ahead heuristic.
    
    Scores candidates by:
      score = dist(current, candidate) + alpha * dist(candidate, destination)
    
    This biases the tour toward the destination, reducing the chance of
    ending up far from the final city.
    """
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to simple nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute distances from current to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from each unvisited to the destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Weighted score: immediate cost + alpha * completion cost
    # alpha = 0.5 balances the two terms; tune for best mean tour length
    alpha = 0.5
    scores = d_current + alpha * d_dest
    
    # Pick the candidate with the lowest score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
