"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin from 150s

def _softmin(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation to min."""
    if len(values) <= 1:
        return values.copy() if len(values) == 1 else np.array([0.0])
    neg_shifted = -values / temperature
    neg_max = np.max(neg_shifted)
    log_sum = neg_max + np.log(np.sum(np.exp(neg_shifted - neg_max)))
    return -temperature * log_sum

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _START_TIME, _TIME_LIMIT
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Convert to numpy array for fast operations
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 1:
        return int(feasible_nodes[0])
    
    # Compute distances
    dist_from_current = distance_matrix[current_node, feasible_nodes].astype(np.float64)
    dist_from_depot = distance_matrix[depot, feasible_nodes].astype(np.float64)
    
    # Compute savings score: maximize dist(depot, next) - dist(current, next)
    savings_score = dist_from_depot - dist_from_current
    
    # Compute smooth regret using soft-min
    # For each node i, soft_regret_i = softmin_j!=i dist_from_current[j] - dist_from_current[i]
    # Use soft-min with a moderate temperature for smoothness
    temperature = 2.0
    soft_regret = np.empty(n_feasible, dtype=np.float64)
    
    for i in range(n_feasible):
        # Get distances excluding node i
        if n_feasible == 2:
            other_idx = 1 - i
            soft_min_excl = dist_from_current[other_idx]
        else:
            mask = np.ones(n_feasible, dtype=bool)
            mask[i] = False
            soft_min_excl = _softmin(dist_from_current[mask], temperature=temperature)
        soft_regret[i] = soft_min_excl - dist_from_current[i]
    
    # Normalize soft regret to [0, 1]
    max_regret = np.max(soft_regret)
    min_regret = np.min(soft_regret)
    if max_regret > min_regret:
        regret_normalized = (soft_regret - min_regret) / (max_regret - min_regret)
    else:
        regret_normalized = np.zeros(n_feasible)
    
    # Normalize savings to [0, 1]
    max_savings = np.max(savings_score)
    min_savings = np.min(savings_score)
    if max_savings > min_savings:
        savings_normalized = (savings_score - min_savings) / (max_savings - min_savings)
    else:
        savings_normalized = np.zeros(n_feasible)
    
    # Return-to-depot cost: penalize nodes far from depot (affects final leg)
    max_depot_dist = np.max(dist_from_depot) if len(dist_from_depot) > 0 else 1.0
    if max_depot_dist > 0:
        depot_penalty_normalized = dist_from_depot / max_depot_dist
    else:
        depot_penalty_normalized = np.zeros(n_feasible)
    
    # Dynamic weighting based on capacity ratio
    total_capacity = 50
    capacity_ratio = rest_capacity / total_capacity
    
    # Smooth weight transition: higher regret weight when capacity is low
    # Use a smooth sigmoid-like transition around capacity_ratio = 0.3
    transition_center = 0.3
    transition_width = 0.2
    if capacity_ratio < transition_center:
        t = (transition_center - capacity_ratio) / transition_width
        regret_weight = 0.25 + 0.35 * min(t, 1.0)
    else:
        regret_weight = 0.25
    savings_weight = 1.0 - regret_weight
    
    # Weight for depot return penalty: stronger when few nodes remain or capacity low
    depot_weight = 0.15 if capacity_ratio < 0.4 else 0.10
    
    # Combined score
    combined_score = (savings_weight * savings_normalized + 
                      regret_weight * regret_normalized - 
                      depot_weight * depot_penalty_normalized)
    
    # We want to maximize the combined score
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
