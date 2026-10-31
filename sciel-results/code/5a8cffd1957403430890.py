import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 1-step lookahead heuristic with destination bias.
    
    Args:
    current_node: int, index of the current city.
    destination_node: int, index of the destination (start) city.
    unvisited_nodes: np.ndarray, array of indices of unvisited cities.
    distance_matrix: np.ndarray, NxN matrix of distances.
    
    Returns:
    int, index of the next city to visit.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen per contract, but safe guard
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        u1, u2 = unvisited_nodes
        # Cost of path c -> u1 -> u2 -> dest
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        # Cost of path c -> u2 -> u1 -> dest
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost1 <= cost2:
            return int(u1)
        else:
            return int(u2)

    # General case: n_unvisited > 2
    # 1. Get distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Calculate nearest neighbor distance for each candidate within the remaining unvisited set
    # For each u_i, we need min(d(u_i, u_j)) for j != i
    # This is O(N^2)
    # We can do this by taking the row of distance_matrix for each u_i, 
    # masking out u_i itself, and taking the min.
    
    # To optimize, we can pre-compute a matrix of min-distances? 
    # Or just loop, as N=50 is small.
    
    min_next_dists = np.zeros(n_unvisited)
    for i, u in enumerate(unvisited_nodes):
        # Get distances from u to all unvisited
        d_u = distance_matrix[u, unvisited_nodes]
        # Mask out the self distance (which is 0)
        # d_u[i] corresponds to distance(u, u) = 0
        # We want min of d_u excluding index i
        # Since d_u[i] is 0, and distances are non-negative, 
        # we can set d_u[i] to a large number and take min
        d_u_copy = d_u.copy()
        d_u_copy[i] = np.inf
        min_next_dists[i] = np.min(d_u_copy)
        
    # 3. Destination bias
    # Bias increases as we get closer to the end
    # Let's use a bias that is 0 for first half, and increases linearly
    # Or simply a constant bias? 
    # A constant bias might hurt early on.
    # Let's try: bias_weight = max(0, 1 - n_unvisited / 20)
    # If n_unvisited > 20, weight is 0.
    # If n_unvisited = 10, weight is 0.5.
    # If n_unvisited = 1, weight is 0.95 (but handled above).
    
    bias_weight = max(0.0, 1.0 - (n_unvisited / 20.0))
    
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Total score: d_curr + min_next_dists + bias_weight * d_dest
    # We might want to weight min_next_dists slightly less than d_curr?
    # Or keep it 1:1.
    scores = d_curr + min_next_dists + bias_weight * d_dest
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
