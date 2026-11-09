"""CVRP heuristic: direct route-cost minimization with savings-aware tie-breaking."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_cache[key]

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > 145

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
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
    
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible, dtype=np.int64)
    n = len(feasible_arr)
    
    # Key distances
    dists_to_current = dm[current_node, feasible_arr]
    dists_to_depot = dm[feasible_arr, depot]
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)]
    
    # For each candidate i, estimate the marginal cost of inserting i into the route:
    # If we go current -> i -> (best next) -> ... -> depot
    # vs current -> (best next) -> ... -> depot
    #
    # Simplified: score[i] = d(current,i) + min_{j != i} [d(i,j) + d(j,depot)]
    # This is a 2-step lookahead that directly estimates route length contribution
    if n > 1:
        # cost[i,j] = d(i,j) + d(j,depot) for j != i
        cost_ij_jd = sub_dm + dists_to_depot[None, :]
        np.fill_diagonal(cost_ij_jd, np.inf)
        best_next_cost = np.min(cost_ij_jd, axis=1)
    else:
        best_next_cost = np.array([0.0])
    
    # Total estimated route contribution if we pick i next:
    # d(current,i) + best_next_cost[i]
    # But this doesn't account for the fact that picking i removes it from future options
    #
    # Alternative: use savings to identify which nodes cluster well together
    # S[i,j] = d(depot,i) + d(depot,j) - d(i,j)
    # High savings pairs should be visited in the same route segment
    savings = dists_to_depot[:, None] + dists_to_depot[None, :] - sub_dm
    # For each node i, its average savings with other nodes
    avg_savings = np.mean(savings, axis=1)
    
    # Composite score: direct cost + penalty for low savings (isolated nodes)
    # We want to minimize: d(current,i) + best_next_cost[i] - lambda * avg_savings[i]
    #
    # Tune lambda based on how many nodes remain:
    # More nodes -> more opportunity for clustering -> higher lambda
    # Fewer nodes -> less benefit from clustering -> lower lambda
    if n > 30:
        lam = 0.5
    elif n > 15:
        lam = 0.3
    elif n > 8:
        lam = 0.15
    else:
        lam = 0.05
    
    scores = dists_to_current + best_next_cost - lam * avg_savings
    
    # Capacity-aware adjustment: when capacity is tight, prefer heavier nodes
    # because they're harder to fit later
    demands_arr = np.array([demands[i] for i in feasible], dtype=np.float64)
    total_demand = np.sum(demands_arr)
    if total_demand > 0:
        util = rest_capacity / total_demand
    else:
        util = 1.0
    
    if util < 0.4:
        # Tight: add small penalty for light nodes (prefer heavy)
        max_d = np.max(demands_arr) if np.max(demands_arr) > 0 else 1.0
        scores -= 0.1 * (demands_arr / max_d)
    
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
