"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Capacity-aware Nearest Neighbor with Depot-Aware 2-Opt and Or-opt.
    
    1. Construct a route segment from current_node using Nearest Neighbor.
    2. Optimize the segment using 2-Opt (including return to depot cost) and Or-opt.
    3. Return the first node of the optimized segment.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard: if we are running out of time, fall back to simple nearest
    if _time_exceeded():
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    dm = distance_matrix
    n = len(feasible)
    feas_arr = np.array(feasible)
    
    # 1. Construct initial route using Capacity-Aware Nearest Neighbor
    remaining_cap = rest_capacity
    unvisited_in_seg = set(feasible)
    route = [current_node]
    
    while len(unvisited_in_seg) > 0:
        # Find nearest neighbor that fits
        u_arr = np.array(list(unvisited_in_seg))
        dists = dm[current_node, u_arr]
        
        # Filter by demand
        valid_mask = demands[u_arr] <= remaining_cap
        if not np.any(valid_mask):
            break
            
        valid_u = u_arr[valid_mask]
        valid_dists = dists[valid_mask]
        
        min_idx = np.argmin(valid_dists)
        best_node = valid_u[min_idx]
        
        # Add to route
        route.append(best_node)
        remaining_cap -= demands[best_node]
        unvisited_in_seg.remove(best_node)
        
        # Move current to best_node
        current_node = best_node
        
        if _time_exceeded():
            if len(route) > 1:
                return int(route[1])
            else:
                return depot

    if len(route) < 3:
        if len(route) > 1:
            return int(route[1])
        return depot

    # 2. Apply Depot-Aware 2-Opt and Or-opt
    # The route is a path: current_node -> ... -> last_node
    # Cost includes return to depot: sum(d[r[i], r[i+1]]) + d[r[-1], depot]
    
    def get_route_cost(r):
        if len(r) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(r) - 1):
            cost += dm[r[i], r[i+1]]
        cost += dm[r[-1], depot]
        return cost

    # 2-Opt Implementation with Depot Return
    improved = True
    max_iter = 10
    iter_count = 0
    n_route = len(route)
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        for i in range(n_route - 2):
            if _time_exceeded():
                if len(route) > 1: return int(route[1])
                return depot
            
            node_i = route[i]
            node_i1 = route[i+1]
            
            for j in range(i + 2, n_route):
                node_j = route[j]
                
                # Calculate old and new costs considering depot return
                # Old edges: (node_i, node_i1) and (node_j, next_j)
                # New edges: (node_i, node_j) and (node_i1, next_j)
                
                if j < n_route - 1:
                    node_j1 = route[j+1]
                    old_len = dm[node_i, node_i1] + dm[node_j, node_j1]
                    new_len = dm[node_i, node_j] + dm[node_i1, node_j1]
                else:
                    # j is the last node, so the edge is (node_j, depot)
                    old_len = dm[node_i, node_i1] + dm[node_j, depot]
                    new_len = dm[node_i, node_j] + dm[node_i1, depot]
                    
                if new_len < old_len - 1e-10:
                    route[i+1:j+1] = reversed(route[i+1:j+1])
                    improved = True
                    if _time_exceeded():
                        if len(route) > 1: return int(route[1])
                        return depot

    # Or-opt Implementation
    # Try moving a single node k to position m
    improved = True
    max_iter = 5
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        n_route = len(route)
        
        for k in range(1, n_route): # Don't move the first node (current_node)
            if _time_exceeded():
                if len(route) > 1: return int(route[1])
                return depot
                
            node_k = route[k]
            
            # Try inserting node_k at every other position m
            # We remove node_k and try inserting it at m
            # m can be from 1 to n_route-1 (index in the reduced list)
            
            # To avoid complex index shifting, we simulate the cost change
            # Removing node_k affects:
            # (route[k-1], route[k]) and (route[k], route[k+1])
            # Inserting at m affects:
            # (route[m-1], route[m]) -> (route[m-1], node_k, route[m])
            
            # We iterate m from 1 to len(route)-1
            # Note: if m == k, no change.
            
            for m in range(1, n_route):
                if m == k:
                    continue
                
                # Determine neighbors in the original route
                # Left neighbor of k: L = route[k-1]
                # Right neighbor of k: R = route[k+1] (if exists, else depot)
                
                L = route[k-1]
                if k < n_route - 1:
                    R = route[k+1]
                    R_is_depot = False
                else:
                    R = depot
                    R_is_depot = True
                
                # Calculate cost of removing k
                # Old: L->k + k->R
                # New: L->R
                cost_remove = dm[L, node_k] + dm[node_k, R]
                cost_connect = dm[L, R]
                
                # Calculate cost of inserting k at m
                # We need to know the neighbors at position m in the route *without* k
                # This is tricky to do efficiently without modifying the list.
                # Given the small size (<= 15 nodes), we can just modify a copy.
                
                # Optimization: Only try if potential gain is high?
                # For now, simple copy-and-check for small n.
                
                route_copy = route[:]
                route_copy.pop(k)
                
                # Adjust m index if m > k
                insert_idx = m
                if m > k:
                    insert_idx = m - 1
                    
                # Bounds check for insert_idx in route_copy
                if insert_idx < 1 or insert_idx > len(route_copy) - 1:
                    continue
                
                # Insert node_k at insert_idx
                # Neighbors in route_copy:
                # Left: route_copy[insert_idx - 1]
                # Right: route_copy[insert_idx]
                
                L_ins = route_copy[insert_idx - 1]
                R_ins = route_copy[insert_idx]
                
                # Cost of inserting:
                # Old: L_ins -> R_ins
                # New: L_ins -> node_k -> R_ins
                cost_insert_old = dm[L_ins, R_ins]
                cost_insert_new = dm[L_ins, node_k] + dm[node_k, R_ins]
                
                # Total change in cost:
                # Delta = (cost_connect - cost_remove) + (cost_insert_new - cost_insert_old)
                delta = (cost_connect - cost_remove) + (cost_insert_new - cost_insert_old)
                
                if delta < -1e-10:
                    # Perform the move
                    route = route_copy
                    route.insert(insert_idx, node_k)
                    improved = True
                    break # Restart loop for this k or break entirely?
                    # Breaking inner loop to restart outer with new route is safer
                    # But we can just break the m loop and continue k loop?
                    # Better to break both and restart while loop to be safe.
            
            if improved:
                break # Restart the while loop

    if _time_exceeded():
        if len(route) > 1: return int(route[1])
        return depot

    if len(route) > 1:
        return int(route[1])
    else:
        return depot

# EVOLVE-BLOCK-END
