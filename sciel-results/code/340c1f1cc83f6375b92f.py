"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a combined heuristic that balances proximity to the current
    node, proximity to the destination, and a penalty for nodes that would create long
    final segments. This is a modified nearest-neighbor with destination awareness.
    
    The score for each candidate node u is:
        score(u) = dist(current, u) + alpha * dist(u, destination) + beta * min_dist(u, unvisited_others)
    
    We use alpha and beta to balance the immediate step cost, the cost to finish at the
    destination, and a penalty for choosing a node that is isolated (far from other unvisited
    nodes), which could lead to a long final leg.
    
    For a 50-city TSP, this should provide a better tour than pure nearest-neighbor by
    avoiding poor choices that lead to expensive final segments.
    """
    if len(unvisited_nodes) == 0:
        # Should not happen, but return destination as fallback
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited node to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For the "isolation" penalty, we want the minimum distance from each unvisited node
    # to any other unvisited node. This is O(n^2) but n is at most 50, so it's fine.
    # Create a submatrix of distances between unvisited nodes
    unvisited_idx = unvisited_nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # For each node, find the min distance to other unvisited nodes (excluding itself)
    # Set diagonal to inf so we don't pick the node itself
    np.fill_diagonal(submatrix, np.inf)
    min_dist_to_other = np.min(submatrix, axis=1)
    
    # Dynamic weighting: prioritize destination more as fewer nodes remain
    n_unvisited = len(unvisited_nodes)
    # alpha ranges from 0.5 (start) to 1.0 (end)
    alpha = 0.5 + 0.5 * (1.0 - n_unvisited / 50.0)
    beta = 0.3    # weight for isolation penalty
    
    # Compute the combined score
    scores = dist_from_current + alpha * dist_to_dest + beta * min_dist_to_other
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
