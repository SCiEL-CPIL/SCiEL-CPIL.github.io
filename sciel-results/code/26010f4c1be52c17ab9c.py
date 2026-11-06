"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using penalty-aware lookahead:
    score = d(current, candidate) + alpha * min(d(candidate, u) for u in unvisited\\{candidate}) 
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
    # This is O(n^2) but n <= 50, so at most 2500 ops
    min_next_hop = np.zeros(n_unvisited)
    for i in range(n_unvisited):
        cand = unvisited_nodes[i]
        # Exclude self from the min
        if n_unvisited > 1:
            # Create mask to exclude self
            mask = np.ones(n_unvisited, dtype=bool)
            mask[i] = False
            min_next_hop[i] = np.min(distance_matrix[cand, unvisited_nodes[mask]])
        else:
            min_next_hop[i] = 0.0
    
    # Penalty term 2: cost from candidate to destination (closure penalty)
    dest_costs = distance_matrix[unvisited_nodes, destination_node]
    
    # Combined score: immediate + alpha * next_hop + beta * dest_closure
    # Tuning alpha and beta: we want lookahead to be meaningful but not dominate
    alpha = 0.3
    beta = 0.5
    
    scores = immediate_costs + alpha * min_next_hop + beta * dest_costs
    
    # Pick the candidate with minimum combined score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
