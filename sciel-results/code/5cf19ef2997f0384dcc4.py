"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic.
    For each candidate u, score = d(current, u) + min_{v in unvisited\{u}} d(u, v).
    If u is the last node, score = d(current, u) + d(u, destination).
    """
    global _start_time, _DEADLINE
    
    # Time limit guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to Nearest Neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1 # Should not happen based on contract, but safe
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Vectorized 2-step lookahead
    # 1. Distance from current to all candidates
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each candidate u, find the min distance to other unvisited nodes
    # d_u_to_others[u_idx, v_idx] = distance_matrix[unvisited_nodes[u_idx], unvisited_nodes[v_idx]]
    # We want min over v != u for each u.
    
    # Get the submatrix of distances between unvisited nodes
    # This is an N x N matrix
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # To find the min excluding self, we can set the diagonal to infinity
    # or simply take the min of the row, but if the min is self (which is 0), we need the 2nd min.
    # Since distance_matrix[i,i] is 0, the min of the row is always 0 (self).
    # We need the min of the row excluding the diagonal element.
    
    # Efficient way:
    # Set diagonal to inf
    np.fill_diagonal(sub_matrix, np.inf)
    
    # min distance from each u to any other unvisited v
    d_next_min = np.min(sub_matrix, axis=1)
    
    # Special case: if there are only 2 unvisited nodes, the "other" is the other one.
    # If n_unvisited == 2, sub_matrix is 2x2. 
    # u0 -> u1, u1 -> u0.
    # This works.
    
    # What if the "next" step should consider the destination?
    # The 2-step lookahead usually looks at the *next* node in the tour.
    # If we pick u, the next node will be chosen by the same heuristic.
    # So d_next_min is a good estimate of the "future cost" immediately following u.
    
    # Total score
    scores = d_current + d_next_min
    
    # Pick the candidate with the lowest score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
