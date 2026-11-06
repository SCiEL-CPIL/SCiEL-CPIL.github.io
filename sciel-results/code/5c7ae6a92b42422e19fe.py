"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for the entire evaluation run to enforce deadline
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: keep 5s margin before 150s limit
    if time.time() - _start_time > 145:
        # Fallback to simple nearest feasible to avoid timeout
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # 1. Identify feasible nodes
    unvisited_array = np.array(unvisited_nodes)
    if len(unvisited_array) == 0:
        return depot
        
    feasible_mask = demands[unvisited_array] <= rest_capacity
    feasible_nodes = unvisited_array[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    n_feasible = len(feasible_nodes)
    
    # 2. Compute immediate distances
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Clarke-Wright savings
    d_i_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    savings = d_i_depot + d_depot_j - dists
    
    # 3. 2-step lookahead for regret
    sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    best_next_costs = np.zeros(n_feasible)
    second_best_next_costs = np.zeros(n_feasible)
    
    for j_idx in range(n_feasible):
        j_node = feasible_nodes[j_idx]
        remaining_cap_after_j = rest_capacity - demands[j_node]
        
        feasible_after_mask = demands[feasible_nodes] <= remaining_cap_after_j
        feasible_after_mask[j_idx] = False
        
        if not np.any(feasible_after_mask):
            best_next_costs[j_idx] = distance_matrix[j_node, depot]
            second_best_next_costs[j_idx] = distance_matrix[j_node, depot]
        else:
            next_dists = sub_dist[j_idx, feasible_after_mask]
            sorted_dists = np.sort(next_dists)
            best_next_costs[j_idx] = sorted_dists[0]
            if len(sorted_dists) > 1:
                second_best_next_costs[j_idx] = sorted_dists[1]
            else:
                second_best_next_costs[j_idx] = sorted_dists[0]
    
    regrets = second_best_next_costs - best_next_costs
    
    # 4. Spatial density scoring: for each candidate j, measure how much
    #    remaining feasible demand clusters around j within a tight radius
    #    This encourages visiting nodes that allow capacity exhaustion in a
    #    localized geographic region
    
    density_scores = np.zeros(n_feasible)
    
    for j_idx in range(n_feasible):
        j_node = feasible_nodes[j_idx]
        remaining_cap = rest_capacity - demands[j_node]
        
        if remaining_cap <= 0:
            density_scores[j_idx] = dists[j_idx]
            continue
        
        # Find nodes that can still be visited after j
        can_visit_mask = demands[feasible_nodes] <= remaining_cap
        can_visit_mask[j_idx] = False
        
        if not np.any(can_visit_mask):
            density_scores[j_idx] = dists[j_idx]
            continue
        
        # Get distances from j to all other feasible nodes
        dists_from_j = distance_matrix[j_node, feasible_nodes]
        
        # Consider only nodes that can be visited after j
        candidate_indices = np.where(can_visit_mask)[0]
        candidate_dists = dists_from_j[candidate_indices]
        
        # Sort candidates by distance
        sorted_order = np.argsort(candidate_dists)
        sorted_cand_indices = candidate_indices[sorted_order]
        sorted_cand_dists = candidate_dists[sorted_order]
        
        # Greedily add nodes within capacity, tracking cluster tightness
        total_demand_added = 0.0
        total_dist_added = 0.0
        count_added = 0
        
        for idx in sorted_cand_indices:
            node = feasible_nodes[idx]
            demand = demands[node]
            dist = dists_from_j[node]
            
            if total_demand_added + demand > remaining_cap:
                break
            
            total_demand_added += demand
            total_dist_added += dist
            count_added += 1
        
        if count_added == 0:
            density_scores[j_idx] = dists[j_idx]
        else:
            # Average distance to clustered nodes
            avg_dist = total_dist_added / count_added
            
            # Capacity utilization: how much of remaining cap we can fill
            capacity_utilization = total_demand_added / remaining_cap
            
            # Density score: lower is better
            # Tight cluster (low avg_dist) with high capacity fill is preferred
            density_scores[j_idx] = avg_dist / (1.0 + 2.0 * capacity_utilization)
    
    # 5. Normalize scores
    if np.max(regrets) > np.min(regrets):
        regrets_norm = (regrets - np.min(regrets)) / (np.max(regrets) - np.min(regrets))
    else:
        regrets_norm = np.zeros(n_feasible)
    
    if np.max(density_scores) > np.min(density_scores):
        density_norm = (density_scores - np.min(density_scores)) / (np.max(density_scores) - np.min(density_scores))
    else:
        density_norm = np.zeros(n_feasible)
    
    # 6. Combine scores
    # Minimize: immediate_dist + density_cost - regret_benefit
    w_immediate = 1.0
    w_density = 0.4
    w_regret = 0.3
    
    scores = w_immediate * dists + w_density * density_norm - w_regret * regrets_norm
    
    # 7. Select best node
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
