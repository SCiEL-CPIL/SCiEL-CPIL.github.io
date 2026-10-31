"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not be called if no nodes are left, but return -1 as a safe fallback
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited node to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Matrix of distances between unvisited nodes
    # This creates an MxM matrix where M is the number of unvisited nodes
    dist_between_unvisited = distance_matrix[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
    
    # We want the minimum distance from each unvisited node j to any OTHER unvisited node k.
    # Set the diagonal to infinity so j is not considered as its own neighbor.
    np.fill_diagonal(dist_between_unvisited, np.inf)
    
    # min_exit_cost[j] is the min distance from unvisited_nodes[j] to any other unvisited node
    min_exit_cost = np.min(dist_between_unvisited, axis=1)
    
    # Weighting: as we get closer to the end of the tour, the destination becomes more important.
    # w_dest increases as n_unvisited decreases.
    w_dest = 1.0 / n_unvisited
    w_exit = 1.0 - w_dest
    
    # Total score for each candidate:
    # Cost to enter + weighted cost to exit to next unvisited + weighted cost to exit to destination
    scores = dist_from_current + w_exit * min_exit_cost + w_dest * dist_to_dest
    
    # Pick the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
