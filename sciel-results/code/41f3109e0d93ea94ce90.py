"""Rewritten CVRP heuristic with improved savings + lookahead scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Safety margin of 5 seconds
    return (time.time() - _start_time) > 145

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Composite savings-based heuristic with multi-step lookahead.
    Combines:
    1. Immediate distance cost from current node
    2. Savings potential (clustering benefit)
    3. Lookahead: estimated future routing cost
    4. Capacity-aware: prefer heavier nodes when remaining capacity is tight
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: fall back to nearest feasible
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
    
    feasible_arr = np.array(feasible, dtype=np.int64)
    n = len(feasible_arr)
    
    # Compute key distances
    dists_to_current = dm[current_node, feasible_arr]
    dists_to_depot = dm[feasible_arr, depot]
    
    # Sub-matrix of distances among feasible nodes
    sub_dm = dm[np.ix_(feasible_arr, feasible_arr)].copy()
    
    # Savings: S[i,j] = d(depot,i) + d(depot,j) - d(i,j)
    # This measures how much route length we save by visiting i and j together
    # rather than in separate trips
    d_depot = dists_to_depot
    savings_matrix = d_depot[:, None] + d_depot[None, :] - sub_dm
    
    # Total savings for each node with all others
    total_savings = np.sum(savings_matrix, axis=1)
    
    # 2-step lookahead: for each candidate i, estimate cost of i -> best_j -> depot
    # This captures the immediate future routing cost
    if n > 1:
        cost_ij_jd = sub_dm + dists_to_depot[None, :]
        np.fill_diagonal(cost_ij_jd, np.inf)
        best_second_step = np.min(cost_ij_jd, axis=1)
    else:
        best_second_step = np.array([0.0])
    
    # 3-step lookahead (only if we have time and enough nodes)
    if n > 2 and not _check_deadline():
        # For each candidate i, estimate i -> best_j -> best_k -> depot
        # This is more expensive but provides better guidance
        # cost_i_j_k_d[i, j, k] = d(i,j) + d(j,k) + d(k,depot)
        # We approximate by: for each i, find best j, then for that j find best k
        # Simplified: use the best_second_step as a proxy and add depot return
        # Actually, let's do a proper 3-step for better accuracy
        three_step = np.zeros(n)
        for i in range(n):
            # Find best j after i
            costs_j = sub_dm[i, :] + dists_to_depot  # d(i,j) + d(j,depot)
            np.fill_diagonal(costs_j[np.newaxis, :] if False else np.diag(costs_j), np.inf)
            # Simpler: for each i, try each j != i, then find best k != i,j
            best_three = np.inf
            for j in range(n):
                if j == i:
                    continue
                # Cost so far: d(i,j)
                # Now find best k != i, j: d(j,k) + d(k,depot)
                costs_k = sub_dm[j, :] + dists_to_depot
                costs_k[i] = np.inf
                costs_k[j] = np.inf
                if n > 2:
                    best_k_cost = np.min(costs_k)
                else:
                    best_k_cost = 0.0
                total = sub_dm[i, j] + best_k_cost
                if total < best_three:
                    best_three = total
            three_step[i] = best_three
    else:
        three_step = best_second_step  # Fallback to 2-step
    
    # Capacity urgency: if rest_capacity is small relative to remaining demands,
    # prefer heavier nodes to fill the vehicle
    demands_arr = np.array([demands[i] for i in feasible], dtype=np.float64)
    total_remaining_demand = np.sum(demands_arr)
    if total_remaining_demand > 0:
        capacity_utilization = rest_capacity / total_remaining_demand
    else:
        capacity_utilization = 1.0
    
    # Normalize components
    max_savings = np.max(total_savings) if np.max(total_savings) > 0 else 1.0
    norm_savings = total_savings / max_savings
    
    max_dtc = np.max(dists_to_current) if np.max(dists_to_current) > 0 else 1.0
    norm_dtc = dists_to_current / max_dtc
    
    max_bss = np.max(best_second_step) if np.max(best_second_step) > 0 else 1.0
    norm_bss = best_second_step / max_bss
    
    max_ts = np.max(three_step) if np.max(three_step) > 0 else 1.0
    norm_ts = three_step / max_ts
    
    max_demand = np.max(demands_arr) if np.max(demands_arr) > 0 else 1.0
    norm_demand = demands_arr / max_demand
    
    # Composite score:
    # We want to MINIMIZE total route length.
    # 
    # Key insight: Nodes with high savings should be visited early to build clusters.
    # The immediate cost d(current, i) matters, but so does the future cost.
    # 
    # Score = w1 * norm_dtc + w2 * norm_ts - w3 * norm_savings + w4 * capacity_term
    #
    # Capacity term: if capacity is tight, prefer heavier nodes (they're harder to fit later)
    # If capacity is loose, prefer lighter nodes to keep options open
    if capacity_utilization < 0.3:
        # Tight capacity: prefer heavier nodes
        capacity_term = norm_demand
    elif capacity_utilization > 0.7:
        # Loose capacity: prefer lighter nodes
        capacity_term = 1.0 - norm_demand
    else:
        # Neutral
        capacity_term = 0.5 * np.ones(n)
    
    # Adaptive weights based on how many nodes remain
    # Early in the route: emphasize savings (clustering)
    # Late in the route: emphasize immediate cost and depot return
    if n > 20:
        w1, w2, w3, w4 = 0.4, 0.3, 0.8, 0.1
    elif n > 10:
        w1, w2, w3, w4 = 0.5, 0.3, 0.6, 0.1
    elif n > 5:
        w1, w2, w3, w4 = 0.6, 0.3, 0.4, 0.1
    else:
        w1, w2, w3, w4 = 0.7, 0.2, 0.2, 0.1
    
    scores = w1 * norm_dtc + w2 * norm_ts - w3 * norm_savings + w4 * capacity_term
    
    # Pick the feasible node with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
