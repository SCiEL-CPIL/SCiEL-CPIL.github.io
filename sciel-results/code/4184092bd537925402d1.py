"""Rewritten program for TSP constructive heuristic using improved minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using an improved minimax strategy:
    For each candidate in unvisited_nodes, compute a score based on:
    1. The distance from the current node to the candidate (immediate cost).
    2. The maximum distance from the candidate to any other unvisited node (bottleneck cost).
    
    The score is the maximum of these two values. This ensures we do not take
    an excessively long edge immediately, nor do we select a node that is 
    isolated from the rest of the unvisited set (which would create a 
    bottleneck later).
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    unvisited_idx = unvisited_nodes
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_idx]
    
    # Get pairwise distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # For each candidate (row), find the max distance to any other unvisited node
    # Exclude self by setting diagonal to -inf
    np.fill_diagonal(submatrix, -np.inf)
    bottleneck_costs = np.max(submatrix, axis=1)
    
    # Combine immediate cost and bottleneck cost
    # We use the maximum to enforce a hard limit on both aspects
    scores = np.maximum(dist_from_current, bottleneck_costs)
    
    # Choose the candidate with the minimum score
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
