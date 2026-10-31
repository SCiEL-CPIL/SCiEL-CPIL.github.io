import numpy as np
import time

_start_time = time.time()
_TOTAL_LIMIT = 150.0
_MARGIN = 5.0

def _is_deadline_passed():
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    if _is_deadline_passed():
        # Fallback to 1-step NN
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # Extract sub-matrices
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Beta for destination pull
    # As n decreases, we want to prioritize getting back to destination more strongly?
    # Or does the structure change?
    # Let's keep beta constant for now, maybe 0.5 is good for 2-step, maybe 0.3 for 3-step?
    # The 3-step already includes a "w" which is closer to dest.
    # The term is d(u,v) + min_w(d(v,w) + beta*d(w,dest)).
    # If beta is high, w is chosen close to dest.
    
    beta = 0.5
    
    # Step 1: Compute min over w for each v
    # M[v, w] = d(v, w) + beta * d(w, dest)
    # Note: d_sub is d(v, w). d_dest is d(w, dest).
    M = d_sub + beta * d_dest[np.newaxis, :]
    np.fill_diagonal(M, np.inf)
    MinW = np.min(M, axis=1) # MinW[v] is min cost from v to dest via 1 intermediate w
    
    # Step 2: Compute min over v for each u
    # K[u, v] = d(u, v) + MinW[v]
    K = d_sub + MinW[np.newaxis, :]
    np.fill_diagonal(K, np.inf)
    MinV = np.min(K, axis=1) # MinV[u] is min cost from u to dest via 2 intermediates v, w
    
    # Step 3: Total score for each u
    # Score[u] = d(curr, u) + MinV[u]
    scores = d_curr + MinV
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
