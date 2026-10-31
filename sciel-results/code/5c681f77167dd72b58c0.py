"""Rewritten program for TSP constructive heuristic using minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a hybrid strategy:
    Primarily nearest neighbor, with a penalty for candidates that are far from
    the rest of the unvisited nodes (bottleneck risk).
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    unvisited_idx = unvisited_nodes
    n_unvisited = len(unvisited_idx)
    
    # 1. Distance from current node to all unvisited nodes
    d_current = distance_matrix[current_node, unvisited_idx]
    
    # 2. Bottleneck cost: Max distance from each unvisited node to other unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    np.fill_diagonal(submatrix, -np.inf)
    bottleneck_costs = np.max(submatrix, axis=1)
    
    # 3. Combine costs
    # We use a weighted sum. The weight for bottleneck is small to avoid
    # the poor performance of pure minimax, but non-zero to prevent
    # picking nodes that are isolated from the remaining cluster.
    weight_bottleneck = 0.2
    scores = d_current + weight_bottleneck * bottleneck_costs
    
    # 4. Penalize visiting the destination node early if it's in unvisited
    # This ensures we return to the start/destination only at the end
    if destination_node in unvisited_idx:
        dest_mask = (unvisited_idx == destination_node)
        if np.sum(dest_mask) > 0:
            # If there are other nodes to visit, penalize the destination
            if n_unvisited > 1:
                scores[dest_mask] += 1000.0
    
    # 5. Select the node with the minimum score
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
