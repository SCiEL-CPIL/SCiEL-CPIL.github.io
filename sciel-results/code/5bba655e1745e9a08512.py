"""Traveling salesman constructive heuristic with vectorized lookahead scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using vectorized multi-term lookahead:
    score = d(current, cand) 
            + alpha * min(d(cand, u) for u in unvisited\{cand})
            + beta  * d(cand, destination)
    
    All terms are computed with NumPy vectorization for speed.
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
    # Vectorized: distance_matrix[cand_idx, unvisited_nodes] gives n_unvisited x n_unvisited
    # We need to exclude diagonal (self) for each row
    cand_indices = unvisited_nodes  # array of candidate node indices
    sub_matrix = distance_matrix[cand_indices[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
    # Set diagonal to infinity so min excludes self
    np.fill_diagonal(sub_matrix, np.inf)
    min_next_hop = np.min(sub_matrix, axis=1)
    
    # Penalty term 2: cost from candidate to destination (closure penalty)
    dest_costs = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive weights: early construction benefits from more lookahead
    frac_remaining = n_unvisited / 50.0  # assume ~50 cities total
    alpha = 0.3 * frac_remaining + 0.1
    beta = 0.5 * frac_remaining + 0.2
    
    # Combined score
    scores = immediate_costs + alpha * min_next_hop + beta * dest_costs
    
    # Pick the candidate with minimum combined score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
