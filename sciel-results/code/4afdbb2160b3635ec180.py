"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix to avoid repeated conversions
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 10 seconds
    return (time.time() - _start_time) > 140

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Combine continuation cost (from current program) with savings-based
    lookahead (from inspiration 1). The composite score balances:
      1. Immediate distance from current node to candidate
      2. Continuation cost: how expensive it will be to extend the route
      3. Savings potential: how much we save by merging this node with others
         instead of separate trips to depot
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to nearest feasible
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n = len(feasible_arr)
    
    # Compute distances from current node to all feasible nodes
    dists_to_current = dm[current_node, feasible_arr]
    
    # Compute distances from each feasible node to depot
    dists_to_depot = dm[feasible_arr, depot]
    
    # Build sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    # Continuation cost: min distance from candidate to any other feasible node or depot
    sub_dm_copy = sub_dm.copy()
    np.fill_diagonal(sub_dm_copy, np.inf)
    min_to_other = np.min(sub_dm_copy, axis=1)
    continuation = np.minimum(min_to_other, dists_to_depot)
    
    # Savings-based lookahead: for each candidate i, compute max savings
    # with any other feasible node j. savings(i,j) = d(depot,i) + d(depot,j) - d(i,j)
    d_depot = dists_to_depot
    savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
    np.fill_diagonal(savings_matrix, -np.inf)
    max_savings = np.max(savings_matrix, axis=1)
    
    # Normalize savings to be comparable to distances
    max_s = np.max(np.abs(max_savings)) if np.max(np.abs(max_savings)) > 0 else 1.0
    norm_savings = max_savings / max_s
    
    # Composite score:
    # score = dist_to_current + alpha * continuation - delta * norm_savings
    # - dist_to_current: immediate cost
    # - alpha * continuation: estimate of future extension cost
    # - delta * norm_savings: reward for nodes that are easy to merge with others
    alpha = 0.5
    delta = 0.4
    scores = dists_to_current + alpha * continuation - delta * norm_savings
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
