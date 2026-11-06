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
        
    # Deadline guard: 150s limit, keep 5s margin
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
    
    # If only one feasible node, return it immediately
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # 2. Regret-based selection with 2-step lookahead
    # For each candidate node j, compute:
    #   - immediate cost: d(current, j)
    #   - regret: how much worse the 2nd-best next step would be if we pick j
    #   - savings: d(current, depot) + d(depot, j) - d(current, j)
    #   - future demand fit: prefer nodes that leave rest_capacity close to a good match
    
    # Immediate distances from current to each feasible node
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Clarke-Wright savings
    d_i_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    savings = d_i_depot + d_depot_j - dists
    
    # 2-step lookahead: for each candidate j, what's the best next node after visiting j?
    # This captures whether picking j leads to a good continuation
    # We compute for each j: min over k != j of d(j, k) where k is feasible and demand fits after j
    # To keep it O(n^2) which is fine for n<=50
    
    # Get the sub-matrix of distances between feasible nodes
    sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each candidate j, compute the best next step cost (excluding self)
    # After visiting j, remaining capacity is rest_capacity - demands[j]
    # So k is feasible after j if demands[k] <= rest_capacity - demands[j]
    
    best_next_costs = np.zeros(n_feasible)
    second_best_next_costs = np.zeros(n_feasible)
    
    for j_idx in range(n_feasible):
        j_node = feasible_nodes[j_idx]
        remaining_cap_after_j = rest_capacity - demands[j_node]
        
        # Find feasible next nodes after visiting j
        feasible_after_mask = demands[feasible_nodes] <= remaining_cap_after_j
        # Exclude j itself
        feasible_after_mask[j_idx] = False
        
        if not np.any(feasible_after_mask):
            # No next node possible, so the "next cost" is returning to depot
            best_next_costs[j_idx] = distance_matrix[j_node, depot]
            second_best_next_costs[j_idx] = distance_matrix[j_node, depot]
        else:
            next_dists = sub_dist[j_idx, feasible_after_mask]
            # Get sorted distances
            sorted_dists = np.sort(next_dists)
            best_next_costs[j_idx] = sorted_dists[0]
            if len(sorted_dists) > 1:
                second_best_next_costs[j_idx] = sorted_dists[1]
            else:
                second_best_next_costs[j_idx] = sorted_dists[0]
    
    # Regret = second_best - best for the next step
    # High regret means picking j forces a bad next move, so we should pick it now
    regrets = second_best_next_costs - best_next_costs
    
    # 3. Combine scores with route completion probability heuristic
    # For each candidate j, estimate the expected remaining tour cost if we visit j next
    # Use a greedy approximation: sum of distances from depot to remaining nodes,
    # minus savings from connecting them in a greedy chain
    
    # For efficiency, we compute a proxy for residual complexity:
    # - After visiting j, remaining capacity = rest_capacity - demands[j]
    # - Remaining feasible nodes are those with demand <= remaining capacity
    # - Estimate remaining tour cost as sum of min distances from depot to each remaining node
    #   (this is an upper bound on the optimal remaining tour)
    
    remaining_capacities = rest_capacity - demands[feasible_nodes]
    
    # For each j, compute estimated remaining cost
    # This is O(n^2) which is fine for n<=50
    est_remaining_costs = np.zeros(n_feasible)
    
    for j_idx in range(n_feasible):
        remaining_cap = remaining_capacities[j_idx]
        if remaining_cap <= 0:
            est_remaining_costs[j_idx] = distance_matrix[feasible_nodes[j_idx], depot]
            continue
        
        # Find nodes that can still be visited after j
        can_visit_mask = demands[feasible_nodes] <= remaining_cap
        can_visit_mask[j_idx] = False  # exclude j itself
        
        if not np.any(can_visit_mask):
            est_remaining_costs[j_idx] = distance_matrix[feasible_nodes[j_idx], depot]
        else:
            # Greedy chain: start from j, keep adding nearest feasible node
            # until no more nodes fit or we've added all
            current_est = feasible_nodes[j_idx]
            total_cost = 0.0
            visited_set = {j_idx}
            
            while True:
                # Find nearest unvisited node that fits
                candidate_mask = can_visit_mask.copy()
                for v in visited_set:
                    candidate_mask[v] = False
                
                if not np.any(candidate_mask):
                    break
                
                # Find nearest
                cand_dists = distance_matrix[current_est, feasible_nodes[candidate_mask]]
                min_idx_local = np.argmin(cand_dists)
                min_idx_global = np.where(candidate_mask)[0][min_idx_local]
                
                total_cost += cand_dists[min_idx_local]
                current_est = feasible_nodes[min_idx_global]
                visited_set.add(min_idx_global)
            
            # Add return to depot
            total_cost += distance_matrix[current_est, depot]
            est_remaining_costs[j_idx] = total_cost
    
    # Normalize est_remaining_costs to be comparable with other terms
    # Use min-max normalization
    if np.max(est_remaining_costs) > np.min(est_remaining_costs):
        est_remaining_norm = (est_remaining_costs - np.min(est_remaining_costs)) / (np.max(est_remaining_costs) - np.min(est_remaining_costs))
    else:
        est_remaining_norm = np.zeros(n_feasible)
    
    # 4. Combine scores
    # Score = immediate_dist + w1 * best_next_cost - w2 * savings - w3 * regrets + w4 * est_remaining_norm
    w1 = 0.3
    w2 = 1.0
    w3 = 0.6
    w4 = 0.5
    
    scores = dists + w1 * best_next_costs - w2 * savings - w3 * regrets + w4 * est_remaining_costs
    
    # 5. Select best node
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
