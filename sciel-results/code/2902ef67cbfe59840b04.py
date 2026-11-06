"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # If time is running out, fallback to simple NN to ensure we finish
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Get distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Sort unvisited nodes by distance from current to get top K candidates
    # K is small, e.g., 10
    k = min(10, n_unvisited)
    top_k_indices = np.argpartition(dists_from_current, k - 1)[:k]
    top_k_nodes = unvisited_nodes[top_k_indices]
    
    best_node = top_k_nodes[0]
    best_score = float('inf')
    
    # For each candidate, simulate the next step (nearest neighbor from candidate)
    # and calculate the 2-step cost.
    # We also consider the distance from that next node to the destination to avoid
    # straying too far from the closing point, though 2-step local is often sufficient.
    
    for c in top_k_nodes:
        # Distance current -> c
        d1 = distance_matrix[current_node, c]
        
        # Find nearest unvisited node to c, excluding c itself
        # We need to be careful with indices. unvisited_nodes contains c.
        # dists_from_c = distance_matrix[c, unvisited_nodes]
        # We want min dists_from_c where unvisited_nodes != c
        
        # Optimization: Pre-compute or just loop? N=50 is small.
        # But we are in a loop of K=10.
        # Let's just use numpy.
        
        # Mask out c
        mask = unvisited_nodes != c
        # If all are masked (should not happen if n_unvisited > 1), handle edge case
        if not np.any(mask):
            continue
            
        dists_from_c = distance_matrix[c, unvisited_nodes]
        # Set distance to c to inf so it's not picked
        dists_from_c_masked = np.where(mask, dists_from_c, np.inf)
        
        # The next node would be the one with min distance
        next_idx = np.argmin(dists_from_c_masked)
        next_node = unvisited_nodes[next_idx]
        d2 = dists_from_c_masked[next_idx]
        
        # Score: d1 + d2
        # We can add a small penalty for distance to destination of the next node
        # to keep the tour compact towards the end.
        # d3 = distance_matrix[next_node, destination_node]
        # score = d1 + d2 + 0.1 * d3  # Tuning parameter?
        
        # Let's stick to pure 2-step local cost first, as it's a standard improvement.
        score = d1 + d2
        
        if score < best_score:
            best_score = score
            best_node = c

    return int(best_node)
# EVOLVE-BLOCK-END
