# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic based on local density.
    
    The heuristic estimates the cost of the next two edges in the tour:
    1. The edge from the current node to the candidate node.
    2. The edge from the candidate node to its nearest unvisited neighbor.
    
    This "nearest-neighbor gap" metric captures the local geometric context
    by penalizing candidates that lead to isolated next steps, effectively
    performing a short-horizon optimization without full tour simulation.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Vectorized distance from current node to all candidates
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute nearest-neighbor distance for each candidate among unvisited nodes
    # This represents the estimated cost of the next edge after visiting the candidate.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Exclude self-distance by setting diagonal to infinity
    sub_matrix_copy = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_copy, np.inf)
    
    # Find the nearest unvisited neighbor for each candidate
    nearest_unvisited_dist = np.min(sub_matrix_copy, axis=1)
    
    # The total estimated cost for choosing candidate i is the sum of:
    # 1. Cost to reach candidate i from current
    # 2. Cost to leave candidate i to its nearest unvisited neighbor
    # This is a 2-step lookahead that balances immediate cost with local connectivity.
    
    score = dist_from_current + nearest_unvisited_dist
    
    # Pick the candidate with the lowest estimated 2-step cost
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
