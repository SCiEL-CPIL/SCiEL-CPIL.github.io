"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for timing
_start_time = None
_DEADLINE = 120.0  # seconds


def _get_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time + _DEADLINE - 5.0  # 5s safety margin


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Check deadline
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) <= 1:
        return int(unvisited_nodes[0])
    
    # Compute scores: distance from current + alpha * distance to destination
    # This is a simple 1-step lookahead that avoids dead-ends
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive alpha: stronger bias to destination when few nodes remain
    n = len(unvisited_nodes)
    if n <= 5:
        alpha = 1.0
    elif n <= 15:
        alpha = 0.5
    else:
        alpha = 0.3
    
    scores = dist_from_current + alpha * dist_to_dest
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
