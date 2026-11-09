"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Nearest Neighbor heuristic.
    This approach minimizes the immediate step cost, which is a standard
    constructive heuristic for TSP and often outperforms biased lookaheads
    that force convergence to the start node too early.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Calculate distances from the current node to all unvisited nodes
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Pick the candidate with the minimum distance
    best_idx = np.argmin(dists)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
