# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 145s safety margin before 150s kill
    if time.time() - _start_time > 145.0:
        # Fallback to nearest neighbor for safety
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Score: distance from current to candidate + distance from candidate to destination
    # This 1-step lookahead avoids long spikes away from the destination
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    d_to_destination = distance_matrix[destination_node, unvisited_nodes]
    
    combined_score = d_from_current + d_to_destination
    
    return int(unvisited_nodes[np.argmin(combined_score)])
# EVOLVE-BLOCK-END
