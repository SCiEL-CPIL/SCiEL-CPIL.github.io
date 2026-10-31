"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard (safety margin)
    # The task is small, but we respect the contract.
    # No complex state tracking needed for this simple heuristic.
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1 # Should not happen if called correctly, but safe
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # 1. Distance from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each candidate unvisited node, find its nearest neighbor 
    #    among the *other* unvisited nodes.
    #    This captures the "connectivity" or "turn" cost.
    
    # We need a matrix of distances between all unvisited nodes.
    # unvisited_nodes is an array of indices.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity so a node doesn't pick itself as its nearest neighbor
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Find the minimum distance from each unvisited node to any other unvisited node
    min_dist_to_other_unvisited = np.min(sub_matrix, axis=1)
    
    # 3. Combine scores.
    # Heuristic: Immediate distance + penalty for being isolated from the rest.
    # A weight of 0.5 balances immediate proximity with future connectivity.
    # This helps avoid "spikes" where a node is close to current but far from everything else.
    weight = 0.5
    scores = dist_from_current + weight * min_dist_to_other_unvisited
    
    # 4. Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
