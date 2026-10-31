"""Rewritten program for TSP constructive heuristic using minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a minimax strategy:
    For each candidate in unvisited_nodes, compute the maximum distance from that candidate
    to any other unvisited node (or to the destination if it's still unvisited).
    Choose the candidate that minimizes this maximum (bottleneck) distance.
    
    This aims to prevent any single long edge in the final tour by ensuring
    that at each step, we don't leave behind cities that would require very long
    connections later.
    """
    if len(unvisited_nodes) == 0:
        # Should not happen, but return current_node as fallback
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # For each unvisited node, compute the maximum distance to any other unvisited node
    # This is the "bottleneck" cost if we were to connect this node to all others
    # We want to minimize this bottleneck
    
    # Get distances from all unvisited nodes to all unvisited nodes
    # unvisited_nodes is an array of node indices
    # distance_matrix[u, v] gives distance from node u to node v
    
    # Compute pairwise distances among unvisited nodes
    # For efficiency with small sets, we can do this directly
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited <= 1:
        return int(unvisited_nodes[0])
    
    # For each candidate unvisited node i, find the max distance to any other unvisited node
    # This is a bottleneck criterion
    
    # Vectorized approach:
    # Get the submatrix of distances between unvisited nodes
    unvisited_idx = unvisited_nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # For each row (candidate), find the maximum distance to any other unvisited node
    # We want to exclude the diagonal (distance to itself is 0)
    # Set diagonal to -inf so it doesn't affect the max
    np.fill_diagonal(submatrix, -np.inf)
    
    # For each candidate, the bottleneck is the max distance to any other unvisited node
    bottleneck_costs = np.max(submatrix, axis=1)
    
    # Choose the candidate with the minimum bottleneck cost
    best_idx = int(np.argmin(bottleneck_costs))
    
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
