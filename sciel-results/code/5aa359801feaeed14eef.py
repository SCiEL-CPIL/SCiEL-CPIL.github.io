# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time, _DEADLINE
    
    if time.time() - _start_time > _DEADLINE:
        if len(unvisited_nodes) == 0:
            return -1
        d_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(d_current)])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Nearest Neighbor
    d_current = distance_matrix[current_node, unvisited_nodes]
    best_idx = np.argmin(d_current)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
