"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin from 150s

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
    # demands is a list/array. Ensure it's aligned with node indices.
    # The interface implies demands[i] is demand of node i.
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # 2. Calculate a composite score
    # We want to minimize total route length.
    # Heuristics:
    # a) Nearest Neighbor: Minimize distance from current node to next node.
    # b) Savings: Maximize (dist(depot, current) + dist(depot, next) - dist(current, next))
    #    This is the classic Clarke-Wright savings heuristic adapted for sequential choice.
    #    Since dist(depot, current) is fixed, we maximize (dist(depot, next) - dist(current, next)).
    # c) Load balancing: If we have very little capacity left, prefer nodes with high demand? 
    #    Or if we have lots of capacity, prefer nodes that are "expensive" to serve from depot?
    
    # Let's try a weighted combination.
    # dist_current = distance_matrix[current_node, feasible_nodes]
    # dist_depot = distance_matrix[depot, feasible_nodes]
    
    # Savings score: dist_depot - dist_current
    # Higher is better for savings.
    
    # Distance score: dist_current
    # Lower is better for distance.
    
    # Let's normalize scores to [0, 1] to combine them?
    # Or just use a simple linear combination.
    
    # Option 1: Pure Savings
    # score = dist_depot - dist_current
    
    # Option 2: Nearest Neighbor
    # score = -dist_current
    
    # Option 3: Hybrid
    # score = alpha * (dist_depot - dist_current) + (1-alpha) * (-dist_current)
    #       = alpha * dist_depot - alpha * dist_current - (1-alpha) * dist_current
    #       = alpha * dist_depot - dist_current
    
    # So if alpha=1, it's pure savings (dist_depot - dist_current).
    # If alpha=0, it's pure nearest (-dist_current).
    
    # Let's try a dynamic alpha.
    # If rest_capacity is low, we should be careful about going far? Or maybe pick the biggest demand that fits?
    # Actually, if capacity is low, we might be forced to close the route soon.
    # Picking the nearest node keeps us close to depot? No, it keeps us close to current.
    # Picking the node with high dist_depot (far from depot) might be good if we can serve it now.
    
    # Let's stick to a robust heuristic: Savings with a penalty for distance.
    # Minimize: dist(current, next) - lambda * (dist(depot, next) - dist(depot, current))
    # Wait, the savings term is (dist(depot, current) + dist(depot, next) - dist(current, next)).
    # Maximizing savings is equivalent to minimizing: dist(current, next) - dist(depot, next) (since dist(depot, current) is const).
    
    # So minimize: dist(current, next) - dist(depot, next)
    # This is the same as minimizing: - (dist(depot, next) - dist(current, next))
    
    # Compute distances
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Regret-based selection combined with savings heuristic
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 1:
        return int(feasible_nodes[0])
    
    # Compute savings score: maximize dist(depot, next) - dist(current, next)
    savings_score = dist_from_depot - dist_from_current
    
    # Compute regret: for each node, how much worse would it be if we didn't visit it now?
    # Regret = distance to nearest alternative - distance to current node
    # High regret means this node is uniquely well-positioned
    # We compute the second-best distance from current to any other feasible node
    # for each candidate
    
    # For efficiency with small n, compute pairwise distances
    # dist_matrix_feasible[i,j] = distance from feasible_nodes[i] to feasible_nodes[j]
    # But we need distance from current to each feasible node, which we already have
    
    # Regret for node i: how much extra distance would we incur if we skip node i?
    # Approximation: min distance from current to other feasible nodes - dist(current, i)
    # If this is large positive, node i is uniquely close and should be visited now
    
    # Compute minimum distance from current to each feasible node (excluding self)
    # For each node i, regret_i = min_j!=i dist(current, j) - dist(current, i)
    
    # Efficient computation: for each i, find the min dist_from_current excluding i
    min_dist_excluding_self = np.full(n_feasible, np.inf)
    for i in range(n_feasible):
        mask = np.ones(n_feasible, dtype=bool)
        mask[i] = False
        min_dist_excluding_self[i] = np.min(dist_from_current[mask])
    
    # Regret: positive means node i is closer to current than any alternative
    regret = min_dist_excluding_self - dist_from_current
    
    # Normalize regret to [0, 1] for combination
    max_regret = np.max(regret)
    min_regret = np.min(regret)
    if max_regret > min_regret:
        regret_normalized = (regret - min_regret) / (max_regret - min_regret)
    else:
        regret_normalized = np.zeros(n_feasible)
    
    # Normalize savings to [0, 1]
    max_savings = np.max(savings_score)
    min_savings = np.min(savings_score)
    if max_savings > min_savings:
        savings_normalized = (savings_score - min_savings) / (max_savings - min_savings)
    else:
        savings_normalized = np.zeros(n_feasible)
    
    # Dynamic weighting: when capacity is low, weight regret more heavily
    # to lock in uniquely positioned nodes before we're forced to close the route
    total_capacity = 50
    capacity_ratio = rest_capacity / total_capacity
    
    # When capacity is low, prioritize high-regret nodes more
    if capacity_ratio < 0.3:
        regret_weight = 0.4 + 0.3 * (0.3 - capacity_ratio) / 0.3  # 0.4 to 0.7
        savings_weight = 1.0 - regret_weight
    else:
        regret_weight = 0.25
        savings_weight = 0.75
    
    # Combined score
    combined_score = savings_weight * savings_normalized + regret_weight * regret_normalized
    
    # Add a small penalty for very long distances to avoid extreme detours
    max_dist = np.max(dist_from_current) if len(dist_from_current) > 0 else 1.0
    if max_dist > 0:
        dist_penalty = 0.05 * (dist_from_current / max_dist)
        combined_score = combined_score - dist_penalty
    
    # We want to maximize the combined score
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
