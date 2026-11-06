"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return depot if we've exceeded 145s (5s buffer before 150s limit)
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)
    
    if n == 0:
        return depot
    
    # Get distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Get demands of unvisited nodes
    unvisited_demands = demands[unvisited]
    
    # Filter feasible nodes (capacity constraint)
    feasible_mask = unvisited_demands <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return depot
    
    feasible_nodes = unvisited[feasible_indices]
    feasible_dist = dist_from_current[feasible_indices]
    feasible_demands = unvisited_demands[feasible_indices]
    
    # If only one feasible, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute savings-based score for each feasible node
    # Savings of visiting node i now vs. going back to depot:
    # If we go to depot then to i: d[current,depot] + d[depot,i]
    # If we go to i then depot: d[current,i] + d[i,depot]
    # Saving = (d[current,depot] + d[depot,i]) - (d[current,i] + d[i,depot])
    # But we also want to consider that after visiting i, we might visit more nodes.
    # So we use a 2-step lookahead.
    
    d_current_depot = distance_matrix[current_node, depot]
    d_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Base savings: how much shorter is it to go current->i->depot vs current->depot->i
    # Actually, the standard savings is: savings[i] = d[depot,i] + d[depot,j] - d[i,j] for merging
    # Here, we're deciding which node to visit next. The key is:
    # - We want to visit nodes that are "close" to each other (high savings between them)
    # - We want to avoid long detours
    
    # Score: negative distance (prefer close) + savings bonus (prefer nodes that save vs depot)
    # savings_bonus for node i: if we visit i, the "cost" is d[current,i] + d[i,depot]
    # vs. just going to depot: d[current,depot]
    # So the incremental cost of visiting i (and then going to depot) is:
    # d[current,i] + d[i,depot] - d[current,depot]
    # We want to minimize this.
    
    incremental_cost = feasible_dist + d_to_depot - d_current_depot
    
    # 2-step lookahead: for each candidate i, find the best j != i to visit after i
    # and compute the combined cost
    # This is O(n^2) but n is at most 50, so it's fine.
    
    # For speed, let's do a lighter lookahead
    # For each feasible node i, compute:
    # cost_i = d[current, i] + min over j != i of (d[i, j] + d[j, depot]) if feasible, else d[i, depot]
    
    # But we also need to respect capacity for the 2-step.
    # Let's do a simplified 2-step:
    # For each i, find the best next node j (by distance from i) that is also feasible
    # and has demand <= rest_capacity - demand[i]
    
    scores = np.zeros(len(feasible_nodes))
    
    for idx_i in range(len(feasible_nodes)):
        i = feasible_nodes[idx_i]
        d_i = feasible_dist[idx_i]
        d_i_depot = d_to_depot[idx_i]
        rem_cap_after_i = rest_capacity - feasible_demands[idx_i]
        
        # Find best 2nd step
        if rem_cap_after_i <= 0:
            # Can't visit any more after i
            best_2nd = d_i_depot
        else:
            # Find feasible j from i
            all_nodes = np.arange(1, len(demands))  # all customer nodes (1-indexed)
            # Actually, we should only consider unvisited nodes
            # But for speed, let's approximate
            dist_from_i = distance_matrix[i, unvisited]
            unvisited_dem = demands[unvisited]
            
            # Exclude i itself
            mask_j = (unvisited != i) & (unvisited_dem <= rem_cap_after_i)
            if np.any(mask_j):
                best_2nd = np.min(dist_from_i[mask_j] + distance_matrix[unvisited[mask_j], depot])
            else:
                best_2nd = d_i_depot
        
        # Total cost if we pick i: d[current,i] + best_2nd
        # Compare to baseline: d[current, depot] (just go back)
        # We want to minimize total route length, so we score by the incremental cost
        scores[idx_i] = d_i + best_2nd
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
