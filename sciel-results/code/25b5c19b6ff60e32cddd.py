"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: if we're running low on time, fall back to nearest neighbor
    elapsed = time.time() - _get_start_time()
    if elapsed > 115.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen, but safety
        return current_node
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Pick the one that minimizes d(current, cand) + d(cand, dest)
        a, b = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, destination_node]
        return a if cost_a <= cost_b else b
    
    # For larger sets: 2-step lookahead
    # For each candidate, score = d(current, cand) + min(d(cand, other) for other in unvisited, other != cand)
    # This avoids picking a node that leads to a dead-end (high cost to continue)
    
    # Vectorized computation
    # d_current_cand: shape (n_unvisited,)
    d_current_cand = distance_matrix[current_node, unvisited_nodes]
    
    # d_cand_unvisited: shape (n_unvisited, n_unvisited)
    d_cand_unvisited = distance_matrix[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
    
    # Set diagonal to inf so we don't pick the candidate itself
    np.fill_diagonal(d_cand_unvisited, np.inf)
    
    # min_next: shape (n_unvisited,) - cheapest edge from cand to any other unvisited
    min_next = np.min(d_cand_unvisited, axis=1)
    
    # Score: immediate distance + cost to continue
    scores = d_current_cand + min_next
    
    # Additionally, add a small penalty for distance to destination to ensure we don't get stuck far from return
    # But this should be small so it doesn't dominate
    d_cand_dest = distance_matrix[unvisited_nodes, destination_node]
    scores += 0.1 * d_cand_dest
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
