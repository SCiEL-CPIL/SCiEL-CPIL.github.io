# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a refined minimax strategy:
    For each candidate in unvisited_nodes, compute a cost that is the maximum of:
    1. The distance from the current node to the candidate.
    2. The maximum distance from the candidate to any other unvisited node (bottleneck within remaining set).
    
    We choose the candidate that minimizes this composite bottleneck cost.
    This prevents selecting nodes that are either too far from the current path 
    or too isolated from the remaining cluster of unvisited cities.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    unvisited_idx = unvisited_nodes
    
    # 1. Distances from current node to all unvisited nodes
    # d_current[i] = distance_matrix[current_node, unvisited_idx[i]]
    d_current = distance_matrix[current_node, unvisited_idx]
    
    # 2. Pairwise distances among unvisited nodes
    # submatrix[u, v] = distance_matrix[unvisited_idx[u], unvisited_idx[v]]
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # For each candidate i, we want the max distance to any OTHER unvisited node j.
    # We exclude the self-distance (diagonal) by setting it to -inf before max.
    # However, we need to be careful if we modify submatrix in place. 
    # Let's create a copy or use a trick.
    # Since n_unvisited is at most 50, copying is fine.
    submatrix_copy = submatrix.copy()
    np.fill_diagonal(submatrix_copy, -np.inf)
    
    # bottleneck_within[i] = max distance from unvisited_idx[i] to any other unvisited node
    bottleneck_within = np.max(submatrix_copy, axis=1)
    
    # Composite cost: max(distance from current, bottleneck within remaining)
    # This ensures we don't pick a node that creates a very long edge from current,
    # nor a node that leaves very long edges between other unvisited nodes (which will have to be traversed).
    composite_cost = np.maximum(d_current, bottleneck_within)
    
    best_idx = int(np.argmin(composite_cost))
    
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
