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
    
    # Regret-based selection: prioritize nodes that are uniquely well-positioned
    # For each feasible node, compute the distance from current to that node.
    # Then find the second-best alternative (in terms of distance from current).
    # The regret is the difference between the best and second-best distances to alternatives.
    # Nodes with high regret are uniquely positioned and should be visited now.
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 1:
        return int(feasible_nodes[0])
    
    # Get all pairwise distances from current node to feasible nodes
    # dist_from_current[i] is distance from current to feasible_nodes[i]
    
    # For each node i, find the second smallest distance among all feasible nodes
    # This represents the "cost" of not visiting node i now (you'd have to go to the next best)
    
    # Sort distances to find second-best for each node
    # Actually, regret for node i = dist(current, i) - min_{j != i} dist(current, j)
    # Wait, that's not quite right. Let me think again.
    
    # Regret-based approach: 
    # For node i, regret = (best distance from current to any feasible node) - (distance from current to node i)
    # No, that would favor the closest node.
    
    # Better regret definition:
    # For node i, compute the cost of NOT visiting it now.
    # If we don't visit i now, we'll visit some other node j first, then come back to i.
    # The additional cost is approximately dist(current, j) + dist(j, i) - dist(current, i)
    # where j is the best alternative to visit before i.
    
    # Simplified regret: for node i, find the best alternative j (j != i) to visit first.
    # The regret is dist(current, j) + dist(j, i) - dist(current, i)
    # We want to maximize this regret (visit nodes with highest regret first).
    
    # This requires computing dist(j, i) for all pairs, which is O(n^2) per call.
    # For 50 customers, this is manageable.
    
    # Get the submatrix of distances among feasible nodes
    feasible_idx = feasible_nodes
    dist_submatrix = distance_matrix[np.ix_(feasible_idx, feasible_idx)]
    
    # For each node i, find the best alternative j to visit before i
    # Cost of visiting j then i: dist(current, j) + dist(j, i)
    # Cost of visiting i now: dist(current, i)
    # Regret(i) = min_j (dist(current, j) + dist(j, i)) - dist(current, i)
    # We want to maximize regret(i)
    
    # Compute for each i: min over j != i of [dist_from_current[j] + dist_submatrix[j, i]]
    regrets = np.zeros(n_feasible)
    for i in range(n_feasible):
        # For node i, find j that minimizes dist_from_current[j] + dist_submatrix[j, i]
        costs = dist_from_current + dist_submatrix[:, i]
        costs[i] = np.inf  # Exclude self
        min_cost = np.min(costs)
        regrets[i] = min_cost - dist_from_current[i]
    
    # Combine regret with savings
    savings_score = dist_from_depot - dist_from_current
    
    # Normalize regret to [0, 1]
    regret_min = np.min(regrets)
    regret_max = np.max(regrets)
    if regret_max > regret_min:
        regret_norm = (regrets - regret_min) / (regret_max - regret_min)
    else:
        regret_norm = np.zeros(n_feasible)
    
    # Combine: weight regret heavily, use savings as tiebreaker
    # Try: score = regret_norm + 0.1 * normalized_savings
    savings_min = np.min(savings_score)
    savings_max = np.max(savings_score)
    if savings_max > savings_min:
        savings_norm = (savings_score - savings_min) / (savings_max - savings_min)
    else:
        savings_norm = np.zeros(n_feasible)
    
    final_score = regret_norm + 0.15 * savings_norm
    
    best_idx = np.argmax(final_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
