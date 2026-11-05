# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using dynamic regret-minimization.
    
    Score for each candidate = -(edge_cost - mean_edge_cost_of_feasible_set)
    This steers away from outlier-long edges, maintaining uniform segment
    lengths and preventing premature capacity exhaustion on distant nodes.
    
    Additional refinements:
    - 2-step lookahead: penalize candidates whose best continuation is expensive
    - Capacity-aware weighting: when capacity is tight, prefer lower-demand nodes
    - Tie-breaking by proximity to depot (for route closure awareness)
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    
    if len(unvisited) == 0:
        return 0
    
    # Filter to feasible candidates (demand fits in rest_capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    n_candidates = len(feasible_nodes)
    
    # Immediate edge costs from current_node to each feasible candidate
    dist_to_candidates = distance_matrix[current_node, feasible_nodes]
    
    # Mean edge cost of the feasible set (baseline for regret)
    mean_edge_cost = np.mean(dist_to_candidates)
    
    # Core regret score: negative of (cost - mean) => favors below-average cost edges
    # This penalizes outlier-long edges
    regret_scores = mean_edge_cost - dist_to_candidates
    
    # 2-step lookahead: for each candidate, estimate the minimum cost of the next edge
    # This helps avoid choosing a cheap edge that leads to an expensive continuation
    two_step_penalties = np.zeros(n_candidates)
    feasible_demands = demands[feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    for idx in range(n_candidates):
        rem_cap = rest_capacity - feasible_demands[idx]
        
        # Find next feasible nodes among unvisited (excluding current candidate)
        next_feasible_mask = demands[unvisited] <= rem_cap
        # Exclude current candidate
        current_idx_in_unvisited = np.where(unvisited == feasible_nodes[idx])[0][0]
        next_feasible_mask[current_idx_in_unvisited] = False
        
        if np.any(next_feasible_mask):
            next_costs = distance_matrix[feasible_nodes[idx], unvisited[next_feasible_mask]]
            min_next_cost = np.min(next_costs)
        else:
            min_next_cost = dist_to_depot[idx]
        
        two_step_penalties[idx] = min_next_cost
    
    # Normalize two-step penalties to [0, 1] range for weighting
    min_ts = np.min(two_step_penalties)
    max_ts = np.max(two_step_penalties)
    ts_range = max_ts - min_ts
    if ts_range > 1e-10:
        normalized_ts = (two_step_penalties - min_ts) / ts_range
    else:
        normalized_ts = np.zeros(n_candidates)
    
    # Capacity tightness: how much capacity remains relative to total remaining demand
    total_unvisited_demand = np.sum(demands[unvisited])
    
    if total_unvisited_demand > 0 and rest_capacity > 0:
        capacity_ratio = rest_capacity / total_unvisited_demand
        # When capacity is tight (ratio < 1), we should be more careful about demand
        # Weight the demand penalty more heavily when capacity is constrained
        demand_weight = max(0.0, 1.0 - capacity_ratio) * 0.3
    else:
        demand_weight = 0.0
    
    # Demand penalty: prefer lower-demand nodes when capacity is tight
    max_demand = np.max(demands[unvisited])
    if max_demand > 0:
        demand_penalty = feasible_demands / max_demand
    else:
        demand_penalty = np.zeros(n_candidates)
    
    # Depot proximity bonus: slightly prefer nodes closer to depot when few nodes remain
    # This helps close routes efficiently
    n_remaining = len(unvisited)
    if n_remaining <= 5:
        depot_dist = distance_matrix[feasible_nodes, depot]
        min_dd = np.min(depot_dist)
        max_dd = np.max(depot_dist)
        dd_range = max_dd - min_dd
        if dd_range > 1e-10:
            depot_proximity = 1.0 - (depot_dist - min_dd) / dd_range
        else:
            depot_proximity = np.zeros(n_candidates)
        depot_weight = 0.15 * (1.0 - n_remaining / 10.0)
    else:
        depot_proximity = np.zeros(n_candidates)
        depot_weight = 0.0
    
    # Combine scores:
    # 1. Primary: regret score (steer away from long edges)
    # 2. Secondary: 2-step lookahead penalty (avoid bad continuations)
    # 3. Tertiary: demand penalty (when capacity tight)
    # 4. Quaternary: depot proximity (when closing route)
    
    # Normalize regret scores to comparable scale
    min_regret = np.min(regret_scores)
    max_regret = np.max(regret_scores)
    regret_range = max_regret - min_regret
    if regret_range > 1e-10:
        norm_regret = (regret_scores - min_regret) / regret_range
    else:
        norm_regret = np.zeros(n_candidates)
    
    combined_scores = (
        1.0 * norm_regret
        - 0.25 * normalized_ts
        - demand_weight * demand_penalty
        + depot_weight * depot_proximity
    )
    
    best_idx = np.argmax(combined_scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
