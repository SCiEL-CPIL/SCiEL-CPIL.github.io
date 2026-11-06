"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

# Global deadline guard
_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s

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

    # Check time before doing expensive work
    if _time_exceeded():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # For small numbers of unvisited nodes, find the exact best next node
    # by evaluating all permutations of the remaining nodes.
    if n_unvisited <= 7:
        best_cost = float('inf')
        best_first_node = -1
        
        # Generate permutations
        for p in permutations(unvisited_nodes):
            # Path: current -> p[0] -> p[1] -> ... -> p[-1] -> destination
            cost = distance_matrix[current_node, p[0]]
            for i in range(len(p) - 1):
                cost += distance_matrix[p[i], p[i+1]]
            cost += distance_matrix[p[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = p[0]
                
        return int(best_first_node)

    # For larger sets, use a multi-factor scoring heuristic
    U = unvisited_nodes
    m = len(U)
    
    # Distance from current to all candidates
    d_curr = distance_matrix[current_node, U]
    
    # Distance from each candidate to destination (exit cost)
    d_dest = distance_matrix[U, destination_node]
    
    # Distance matrix between all candidates
    sub = distance_matrix[np.ix_(U, U)].copy()
    np.fill_diagonal(sub, np.inf)
    
    # 1-step lookahead: min distance from u to any other v in U
    min_next = np.min(sub, axis=1)
    
    # 2-step lookahead: for each u, min over v,w in U\{u} of d(u,v) + d(v,w)
    # This gives a better estimate of the "local" path cost
    # For each u, we want min_{v != u} (d(u,v) + min_{w != u,v} d(v,w))
    # We can compute this as: for each v, min_w d(v,w) where w != u, then add d(u,v)
    # But this is tricky to vectorize. Let's use a simpler approximation:
    # min_2step[u] = min_v (d(u,v) + min_next[v]) for v != u
    # This is O(m^2) which is fine for m <= 49
    
    min_2step = np.zeros(m)
    for i in range(m):
        # For candidate i, min over j != i of d(i,j) + min_next[j]
        # min_next[j] is already the min distance from j to any other node in U (excluding j itself)
        # But we need min over w != i, j of d(j,w), which is not exactly min_next[j]
        # as min_next[j] might pick w = i. However, this is a reasonable approximation.
        costs = sub[i, :] + min_next
        costs[i] = np.inf  # Exclude self
        min_2step[i] = np.min(costs)
    
    # Adaptive weights based on position in tour
    # Early in tour: weight immediate cost more
    # Late in tour: weight exit cost more
    progress = (50 - n_unvisited) / 50.0  # rough progress estimate
    
    # Weights that shift from immediate to exit cost
    w_curr = 1.0
    w_next = 0.3 * (1.0 - progress)  # less important as we get closer to end
    w_2step = 0.2 * (1.0 - progress)
    w_dest = 0.5 + 0.5 * progress  # more important near end
    
    scores = w_curr * d_curr + w_next * min_next + w_2step * min_2step + w_dest * d_dest
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])
# EVOLVE-BLOCK-END
