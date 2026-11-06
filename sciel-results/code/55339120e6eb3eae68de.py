"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes
    unvisited_demands = demands[unvisited]
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    
    if _time_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # 1. Immediate distance from current
    dists_current = distance_matrix[current_node, feasible_nodes]
    
    # 2. Distance to depot (return cost proxy)
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # 3. 2-step lookahead: minimum distance from each feasible node to any other feasible node
    # This avoids dead-ends where a node is close but leads to a long detour
    feas_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    np.fill_diagonal(feas_dist, np.inf)
    min_next_dist = np.min(feas_dist, axis=1)
    
    # 4. Cluster centrality: mean distance to all unvisited nodes
    # Avoids picking outliers far from the remaining work
    dists_feas_to_unvisited = distance_matrix[np.ix_(feasible_nodes, unvisited)]
    sum_dists = np.sum(dists_feas_to_unvisited, axis=1)
    n_unvisited = len(unvisited)
    if n_unvisited > 1:
        mean_dists = sum_dists / (n_unvisited - 1)
    else:
        mean_dists = np.zeros(n_feasible)
    
    # Combine scores with weights:
    # 1.0 for immediate distance (primary driver)
    # 0.5 for depot distance (keep route tight to origin)
    # 0.3 for 2-step lookahead (avoid dead-ends)
    # 0.2 for cluster centrality (stay in the pack)
    scores = dists_current + 0.5 * dists_depot + 0.3 * min_next_dist + 0.2 * mean_dists
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
