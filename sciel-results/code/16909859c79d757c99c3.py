"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Nearest neighbour with a light lookahead bias toward the destination.
    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Time guard: if we're running long, fall back to pure NN
    t0 = time.time()
    if t0 > 145.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute distances from current to unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from unvisited to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Use a weighted combination: favor nearest neighbors but also consider
    # how close the candidate is to the destination (to avoid long detours)
    # Alpha controls the balance; 0.7 weight on immediate distance, 0.3 on destination proximity
    alpha = 0.7
    scores = alpha * dists_from_current + (1 - alpha) * dists_to_dest
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
