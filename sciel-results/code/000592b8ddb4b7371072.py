import numpy as np
import time

# Global state for deadline guard
_start_time = None
_TIME_LIMIT = 140.0  # Safety margin of 10s

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Check deadline
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple NN to be safe, though unlikely to trigger
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen if called correctly, but handle gracefully
        return -1 
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Get the unvisited node IDs
    nodes = unvisited_nodes
    
    # 1. Distance from current to all candidates
    dist_from_current = distance_matrix[current_node, nodes]
    
    # 2. Compute exit costs: distance from candidate to its nearest neighbor in the remaining set
    # Create submatrix of distances between unvisited nodes
    # nodes is an array of indices
    sub_matrix = distance_matrix[np.ix_(nodes, nodes)]
    
    # We need the min of each row, excluding the diagonal (self-distance)
    # Set diagonal to infinity to ignore self
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Find the minimum distance in each row
    # This is the cost of the "next" step after visiting the candidate
    exit_costs = np.min(sub_matrix, axis=1)
    
    # Total cost: distance to candidate + cost to leave candidate
    total_costs = dist_from_current + exit_costs
    
    # Select the candidate with minimum total cost
    best_idx = np.argmin(total_costs)
    
    return int(nodes[best_idx])
