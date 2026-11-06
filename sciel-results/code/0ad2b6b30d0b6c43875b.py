# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a local density optimization heuristic with lookahead.
    
    The score for each candidate node i is estimated as the incremental cost of
    extending the partial tour from current_node to i and then to the nearest
    unvisited neighbor of i. This avoids full tour simulation and focuses on
    maintaining a compact local frontier of unvisited nodes.
    
    Score(i) = dist(current_node, i) + lambda * dist(i, nearest_unvisited_neighbor)
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Vectorized distance from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute the distance from each unvisited node to its nearest unvisited neighbor
    # We restrict the distance matrix to the unvisited subgraph.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # To find the nearest neighbor excluding self, we set the diagonal to infinity
    # before taking the minimum.
    sub_matrix_copy = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_copy, np.inf)
    
    # The minimum distance to any other unvisited node
    nearest_unvisited_dist = np.min(sub_matrix_copy, axis=1)
    
    # Lookahead weight: balances immediate distance vs. local connectivity
    # A higher weight penalizes isolated nodes more strongly.
    lambda_lookahead = 0.5
    
    score = dist_from_current + lambda_lookahead * nearest_unvisited_dist
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
