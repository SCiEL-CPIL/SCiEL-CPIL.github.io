# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic.
    
    For each candidate node u, we estimate the cost of moving from current to u, 
    and then from u to its best next candidate v. 
    The set of possible next candidates v includes all other unvisited nodes 
    and the destination node.
    
    Score(u) = dist(current, u) + min(dist(u, v) for v in unvisited_others U {destination})
    
    This approach dynamically balances immediate proximity and future connectivity, 
    allowing the tour to "turn" towards the destination earlier if it is geometrically 
    advantageous, rather than relying on fixed weights.
    """
    if len(unvisited_nodes) == 0:
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # We need to calculate the minimum distance from each unvisited node 
    # to any node in the set: (unvisited_nodes \ {u}) U {destination_node}
    
    # To do this efficiently:
    # 1. Get the submatrix of distances between unvisited nodes.
    # 2. Get the vector of distances from unvisited nodes to destination.
    
    unvisited_idx = unvisited_nodes
    n_unvisited = len(unvisited_idx)
    
    # Submatrix of distances between unvisited nodes
    # shape (n_unvisited, n_unvisited)
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # Distances from each unvisited node to the destination
    # shape (n_unvisited,)
    dist_to_dest = distance_matrix[unvisited_idx, destination_node]
    
    # For each node i in unvisited_idx:
    # We want min( min(submatrix[i, j] for j != i), dist_to_dest[i] )
    
    # Step 1: Find min distance to other unvisited nodes (excluding self)
    # Set diagonal to inf
    submatrix_diag_inf = submatrix.copy()
    np.fill_diagonal(submatrix_diag_inf, np.inf)
    
    # Min distance to any other unvisited node
    min_dist_to_other_unvisited = np.min(submatrix_diag_inf, axis=1)
    
    # Step 2: The candidate next node v can also be the destination.
    # So the "best next step" cost is min(min_dist_to_other_unvisited, dist_to_dest)
    best_next_step_cost = np.minimum(min_dist_to_other_unvisited, dist_to_dest)
    
    # Total score
    scores = dist_from_current + best_next_step_cost
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
