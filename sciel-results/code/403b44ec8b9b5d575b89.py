# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        u1 = unvisited_nodes[0]
        u2 = unvisited_nodes[1]
        
        # Cost of c -> u1 -> u2 -> dest
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        # Cost of c -> u2 -> u1 -> dest
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost1 < cost2:
            return int(u1)
        else:
            return int(u2)

    # General case: Vectorized 2-step lookahead with destination awareness
    # Score(u) = dist(current, u) + min_{v != u} (dist(u, v) + dist(v, destination))
    
    u_arr = unvisited_nodes
    n = n_unvisited
    
    # Distances from current to all unvisited
    d_current = distance_matrix[current_node, u_arr]
    
    # Distances from all unvisited to destination
    d_dest = distance_matrix[u_arr, destination_node]
    
    # Distances between all unvisited pairs
    # u_arr is 1D array of indices. distance_matrix is 2D.
    # d_pairs[i, j] = dist(u_arr[i], u_arr[j])
    d_pairs = distance_matrix[np.ix_(u_arr, u_arr)]
    
    # We want for each i: min_j (d_pairs[i, j] + d_dest[j]) for j != i
    # Create a matrix where row i contains d_pairs[i, :] + d_dest
    # d_dest is shape (n,), d_pairs is (n, n).
    # d_pairs + d_dest[None, :] broadcasts d_dest to each row.
    
    tail_costs_matrix = d_pairs + d_dest[None, :]
    
    # Mask the diagonal to infinity so we don't pick u itself as the next node v
    np.fill_diagonal(tail_costs_matrix, np.inf)
    
    # Find the minimum tail cost for each candidate u
    min_tail_costs = np.min(tail_costs_matrix, axis=1)
    
    # Total score for each candidate
    scores = d_current + min_tail_costs
    
    best_idx = np.argmin(scores)
    return int(u_arr[best_idx])
# EVOLVE-BLOCK-END
