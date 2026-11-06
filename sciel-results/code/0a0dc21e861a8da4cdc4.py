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

    # 1. Immediate distance from current
    dists_current = distance_matrix[current_node, feasible_nodes]
    
    # 2. Distance to depot (return cost proxy)
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # 3. Mean distance to all unvisited nodes (cluster centrality proxy)
    # We want to avoid picking nodes that are far from the remaining cluster,
    # as this might lead to inefficient backtracking later.
    # Note: We include ALL unvisited nodes in the mean, not just feasible ones,
    # because the infeasible ones will still need to be visited (in other routes),
    # so the "center of gravity" of all remaining work is relevant.
    
    # Distance matrix from feasible_nodes to ALL unvisited_nodes
    # Shape: (num_feasible, num_unvisited)
    dists_feas_to_unvisited = distance_matrix[np.ix_(feasible_nodes, unvisited)]
    
    # Sum distances to all other unvisited nodes
    # Subtract self distance (which is 0) implicitly by dividing by (N-1)
    # Since diagonal is 0, sum is correct for k != j
    sum_dists = np.sum(dists_feas_to_unvisited, axis=1)
    
    n_unvisited = len(unvisited)
    if n_unvisited > 1:
        mean_dists = sum_dists / (n_unvisited - 1)
    else:
        mean_dists = np.zeros(len(feasible_nodes))
        
    # Combine scores
    # Weights:
    # 1.0 for immediate distance (primary driver)
    # 1.0 for depot distance (keep route tight to origin)
    # 0.5 for cluster centrality (avoid outliers, stay in the pack)
    
    scores = dists_current + 1.0 * dists_depot + 0.5 * mean_dists
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
