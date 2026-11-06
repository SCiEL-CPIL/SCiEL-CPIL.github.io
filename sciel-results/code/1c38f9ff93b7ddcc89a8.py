"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

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
    
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small remaining sets, use exact permutation search
    if n_unvisited <= 8:
        best_cost = float('inf')
        best_first = -1
        U = unvisited_nodes
        for p in permutations(U):
            cost = distance_matrix[current_node, p[0]]
            for i in range(len(p) - 1):
                cost += distance_matrix[p[i], p[i+1]]
            cost += distance_matrix[p[-1], destination_node]
            if cost < best_cost:
                best_cost = cost
                best_first = p[0]
        return int(best_first)
    
    # For larger sets, use vectorized 2-step lookahead with adaptive weighting
    U = unvisited_nodes
    m = n_unvisited
    
    d_curr = distance_matrix[current_node, U]
    d_dest = distance_matrix[U, destination_node]
    
    # 2-step lookahead: for each candidate c, estimate min_{n in U\{c}} (d(c,n) + d(n,dest))
    # Vectorized: build sub-matrix of distances between U nodes
    sub = distance_matrix[np.ix_(U, U)]
    # Set diagonal to inf so we don't pick the node itself
    np.fill_diagonal(sub, np.inf)
    
    # For each candidate i, the 2-step cost is min_j (sub[i,j] + d_dest[j])
    # This can be computed as: for each i, min over j of (sub[i,j] + d_dest[j])
    # Vectorized: add d_dest to each column, then take min over columns
    two_step = np.min(sub + d_dest[np.newaxis, :], axis=1)
    
    # Adaptive weights: shift from immediate cost to exit cost as tour progresses
    # Estimate progress: we've visited (50 - m) nodes out of 50
    progress = max(0.0, (50.0 - m) / 50.0)
    
    # Weight immediate cost heavily early, exit cost heavily late
    w_curr = 1.0
    w_dest = 0.3 + 0.7 * progress
    w_2step = 0.2 * (1.0 - progress)  # 2-step lookahead less important late
    
    scores = w_curr * d_curr + w_2step * two_step + w_dest * d_dest
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])
# EVOLVE-BLOCK-END
