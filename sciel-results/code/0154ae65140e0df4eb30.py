import numpy as np
import time

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

    # If time is critical, fallback to nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Depth 2 Lookahead Heuristic
    # 1. Get distances from current to all unvisited
    dist_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get submatrix of distances between unvisited nodes
    # M[i, j] is distance between unvisited_nodes[i] and unvisited_nodes[j]
    M = distance_matrix[unvisited_nodes, unvisited_nodes]
    
    # 3. Compute nearest neighbor distance for each node in unvisited, excluding itself
    # We need min_{j != i} M[i, j]
    M_temp = M.copy()
    np.fill_diagonal(M_temp, np.inf)
    
    # If n_unvisited is 2, M_temp will have one inf and one valid value per row.
    # nn_dist[i] is the distance from unvisited[i] to its closest other unvisited node
    nn_dist = np.min(M_temp, axis=1)
    
    # 4. Compute the lookahead cost for each candidate u (index i in unvisited)
    # Cost(i) = dist(current, u_i) + min_{j != i} (dist(u_i, u_j) + nn_dist[j])
    
    # Create a matrix where entry (i, j) is dist(u_i, u_j) + nn_dist[j]
    # We add nn_dist to each column
    M_lookahead = M + nn_dist[None, :]
    
    # We cannot pick j == i, so set diagonal to inf
    np.fill_diagonal(M_lookahead, np.inf)
    
    # For each i, find the min over j
    min_next_step_cost = np.min(M_lookahead, axis=1)
    
    # Total score for choosing u_i as next node
    total_score = dist_to_unvisited + min_next_step_cost
    
    # Pick the index with minimum total score
    best_idx = np.argmin(total_score)
    
    return int(unvisited_nodes[best_idx])
