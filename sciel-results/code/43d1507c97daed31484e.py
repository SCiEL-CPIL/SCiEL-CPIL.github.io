"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a greedy lookahead strategy.
    Score = dist(current, candidate) + alpha * dist(candidate, destination)
    This avoids trapping the tour in a dead-end far from the destination.
    """
    if len(unvisited_nodes) == 0:
        return -1 # Should not happen based on contract, but safe guard
        
    # If only one node left, must pick it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # 1. Calculate distance from current to all unvisited candidates
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Calculate distance from all unvisited candidates to the destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Combine scores. 
    # A weight of 0.5 balances immediate proximity with future return cost.
    # This is effectively a 2-step greedy heuristic.
    alpha = 0.5
    scores = dists_from_current + alpha * dists_to_dest
    
    # 4. Pick the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
