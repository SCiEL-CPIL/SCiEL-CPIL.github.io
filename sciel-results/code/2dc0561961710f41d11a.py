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
    
    if _time_exceeded():
        # Fallback to nearest neighbor if time is running out
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Score each candidate: immediate distance + weighted lookahead to destination
    # This is a 1-step lookahead heuristic within the greedy constructive family
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    d_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Use a weighted sum: immediate cost + alpha * exit cost
    # alpha=0.5 balances immediate and future costs
    alpha = 0.5
    scores = d_from_current + alpha * d_to_destination
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
