"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _time_remaining():
    return 145.0 - (time.time() - _get_start_time())  # 5s safety margin

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot

    # Check if we're close to deadline - use fast fallback
    if _time_remaining() < 1.0:
        # Fast nearest feasible neighbor
        unvisited = np.array(unvisited_nodes)
        feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
        if not np.any(feasible_mask):
            return depot
        feasible_nodes = unvisited[feasible_mask]
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    # Convert to numpy array for fast operations
    unvisited = np.array(unvisited_nodes)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    n_feasible = len(feasible_nodes)
    
    # Get distances from current node to feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Get distances from depot to feasible nodes
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    # 2. Calculate Regret-Based Score
    # Regret for node j = dist_to_nearest_other - dist_to_current
    # High regret means this node is uniquely well-positioned and delaying it causes a long detour
    
    if n_feasible > 1:
        # For each feasible node j, find the minimum distance to any other feasible node
        # dist_matrix among feasible nodes
        feas_dist_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        
        # For each node j, the nearest other feasible node distance
        # Set diagonal to inf to exclude self
        np.fill_diagonal(feas_dist_matrix, np.inf)
        nearest_other_dist = np.min(feas_dist_matrix, axis=1)
        
        # Regret = how much worse it would be to connect to nearest other vs current
        # If regret is high, this node has a unique good connection to current
        regret = dist_from_current - nearest_other_dist
    else:
        # Only one feasible node, no regret calculation needed
        regret = np.array([0.0])
    
    # 3. Combine savings and regret
    # Savings = dist_from_depot - dist_from_current (traditional Clarke-Wright)
    savings = dist_from_depot - dist_from_current
    
    # Normalize both to [0, 1] range for combination
    if n_feasible > 1:
        savings_min, savings_max = np.min(savings), np.max(savings)
        regret_min, regret_max = np.min(regret), np.max(regret)
        
        if savings_max > savings_min:
            savings_norm = (savings - savings_min) / (savings_max - savings_min)
        else:
            savings_norm = np.zeros_like(savings)
            
        if regret_max > regret_min:
            regret_norm = (regret - regret_min) / (regret_max - regret_min)
        else:
            regret_norm = np.zeros_like(regret)
    else:
        savings_norm = np.array([1.0])
        regret_norm = np.array([0.0])
    
    # Weighted combination: emphasize regret to avoid future long detours
    # Higher weight on regret means we prioritize nodes that are uniquely well-positioned
    w_regret = 0.6
    w_savings = 0.4
    
    combined_score = w_regret * regret_norm + w_savings * savings_norm
    
    # 4. Select the best node
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
