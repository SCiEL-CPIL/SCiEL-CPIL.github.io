"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using penalty-aware lookahead:
    score = d(current, candidate) + alpha * min(d(candidate, u) for u in unvisited\{candidate}) 
            + beta * d(candidate, destination)
    
    This discourages choices that strand the tour far from the return city.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to pure nearest neighbor
    if time.time() - _start_time > TIME_LIMIT:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Immediate edge costs from current to each candidate
    immediate_costs = distance_matrix[current_node, unvisited_nodes]
    
    # Penalty term 1: min cost from candidate to any other unvisited node (next-hop proxy)
    # Vectorized calculation
    # Create submatrix of distances between unvisited nodes
    # dist_sub[i, j] = d(unvisited[i], unvisited[j])
    dist_sub = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(dist_sub, np.inf)
    min_next_hop = np.min(dist_sub, axis=1)
    
    # Penalty term 2: cost from candidate to destination (closure penalty)
    dest_costs = distance_matrix[unvisited_nodes, destination_node]
    
    # Combined score
    # Tuned weights for better global tour quality
    alpha = 0.5
    beta = 0.5
    
    scores = immediate_costs + alpha * min_next_hop + beta * dest_costs
    
    # Pick the candidate with minimum combined score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
