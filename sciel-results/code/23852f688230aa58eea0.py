"""Combined TSP constructive heuristic: MST-aware 2-step lookahead with adaptive destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 2 seconds before the 120s limit (safety margin)."""
    return _start_time + 118.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Combine three structural signals:
    1. Nearest-neighbor from current (greedy local step)
    2. Adaptive destination bias (global tour-closing awareness)
    3. MST-aware connectivity score (global structure to avoid dead-ends)
    
    The score for candidate c:
      d(current, c) + alpha(n) * d(c, dest) + beta * mst_penalty(c)
    
    where mst_penalty uses the minimum edge connecting c to the rest of unvisited,
    providing a cheap proxy for MST structure without full MST computation.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive alpha: stronger destination bias as fewer nodes remain
    total_nodes = distance_matrix.shape[0]
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    alpha = 0.02 + 0.33 * (progress ** 2)
    
    scores = dists_from_current + alpha * dists_to_dest
    
    # MST-aware connectivity: for each candidate, find the minimum distance
    # to any OTHER unvisited node (excluding itself). This approximates how
    # "connected" the candidate is to the remaining tour, penalizing choices
    # that would create isolated long edges later.
    if n_unvisited >= 3:
        unvisited_idx = unvisited_nodes.astype(int)
        sub_matrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
        
        diag_idx = np.arange(n_unvisited)
        sub_matrix_no_diag = sub_matrix.copy()
        sub_matrix_no_diag[diag_idx, diag_idx] = 1e10
        
        # For each candidate, the minimum edge to any other unvisited node
        # This is a cheap proxy for MST connectivity
        min_outgoing = np.min(sub_matrix_no_diag, axis=1)
        
        # Normalize by mean to make scale-invariant
        mean_min_out = np.mean(min_outgoing)
        if mean_min_out > 1e-10:
            min_outgoing_norm = min_outgoing / mean_min_out
        else:
            min_outgoing_norm = min_outgoing
        
        # Beta: moderate weight for connectivity awareness
        # Early in tour: smaller beta (more freedom), late: larger (avoid dead-ends)
        beta = 0.15 + 0.2 * progress
        
        scores += beta * min_outgoing_norm
    
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
