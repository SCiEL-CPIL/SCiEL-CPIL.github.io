"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Nearest Neighbor heuristic with 2-step lookahead.
    
    This approach evaluates potential next nodes by considering the immediate cost
    to the candidate node plus the cost to its nearest neighbor among the remaining
    unvisited nodes. This helps avoid local minima where a close node leads to a
    distant next stop.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        # Cost of going current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost of going current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2

    # Get the array of unvisited node indices
    unvisited = unvisited_nodes.astype(int)
    
    # Pre-compute distances from current to all unvisited
    d_current_to_unvisited = distance_matrix[current_node, unvisited]
    
    # Compute the sub-matrix of distances between unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited, unvisited)]
    
    # Create a copy to modify the diagonal (exclude self)
    sub_dist_copy = sub_dist.copy()
    np.fill_diagonal(sub_dist_copy, np.inf)
    
    # Find the nearest neighbor distance for each unvisited node
    # min_neighbor_dists[i] is the minimum distance from unvisited[i] to any other unvisited node
    min_neighbor_dists = np.min(sub_dist_copy, axis=1)
    
    # Total lookahead score: cost to reach node + cost to its nearest neighbor
    scores = d_current_to_unvisited + min_neighbor_dists
    
    # Pick the index with the minimum score
    best_idx = np.argmin(scores)
    best_node = int(unvisited[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
