"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # If no unvisited nodes, this shouldn't happen, but handle gracefully
    if len(unvisited_nodes) == 0:
        return -1
        
    # Calculate scores for each unvisited node
    # Score is the sum of distance from current to node and from node to destination
    # This encourages closing the tour loop efficiently
    current_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    unvisited_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    scores = current_to_unvisited + unvisited_to_dest
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    return int(unvisited_nodes[min_idx])
# EVOLVE-BLOCK-END
