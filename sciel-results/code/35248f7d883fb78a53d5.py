"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)
    
    if n == 0:
        return int(depot)
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Get demands of unvisited nodes
    unvisited_demands = demands[unvisited]
    
    # Feasibility: demand must fit in remaining capacity
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    feasible_dist = dist_from_current[feasible_indices]
    feasible_demands = unvisited_demands[feasible_indices]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # Compute distance from each feasible candidate to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Regret-based selection: for each candidate, compute the regret
    # Regret = (second best option cost) - (best option cost)
    # High regret means this node is uniquely well-positioned
    
    m = len(feasible_nodes)
    
    if m == 1:
        return int(feasible_nodes[0])
    
    # Compute pairwise distances between feasible nodes
    # dist_matrix[i,j] = distance from feasible_nodes[i] to feasible_nodes[j]
    feasible_dist_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each node i, compute the savings of visiting i next vs. going to depot
    # Savings = dist(current, i) + dist(i, depot) - dist(current, depot)
    # If savings is high, visiting i next is beneficial
    
    dist_current_depot = distance_matrix[current_node, depot]
    savings = feasible_dist + dist_to_depot - dist_current_depot
    
    # Regret computation:
    # For each node i, find the best alternative (excluding i)
    # The regret is how much worse the best alternative is compared to i
    # We approximate: for node i, consider going to depot after i vs. going to another node j after i
    
    # Simplified regret: for each node i, compute the cost if we DON'T visit i next
    # (i.e., visit some other node j first, then i) vs. visiting i now
    
    # Cost if visit i now: dist(current, i) + [future cost from i]
    # Cost if visit j first then i: dist(current, j) + dist(j, i) + [future cost from i]
    # The difference is: dist(current, i) - dist(current, j) - dist(j, i)
    # If this is negative for all j, then visiting i now is better
    
    # Regret[i] = min over j != i of [dist(current, j) + dist(j, i) - dist(current, i)]
    # If regret[i] is large positive, it means even the best alternative j is much worse than going to i
    
    # Actually, let's use a simpler regret:
    # For each node i, the "opportunity cost" of not visiting it is the extra distance
    # incurred by visiting some other node first
    
    # Compute for each pair (i, j): extra_cost[i,j] = dist(current,j) + dist(j,i) - dist(current,i)
    # This is the extra cost of going to j before i instead of i directly
    
    # Regret[i] = min over j != i of extra_cost[i,j]
    # High regret means i is uniquely well-positioned
    
    # Vectorized computation
    # extra_cost[i,j] = feasible_dist[j] + feasible_dist_matrix[j,i] - feasible_dist[i]
    
    # For each i, find min over j != i
    # Create matrix where row i, col j = feasible_dist[j] + feasible_dist_matrix[j,i] - feasible_dist[i]
    
    # feasible_dist[j] for all j: shape (m,)
    # feasible_dist_matrix[j,i] for all j: this is column i of feasible_dist_matrix, shape (m,)
    # feasible_dist[i]: scalar for each i
    
    # extra_cost_matrix[i,j] = feasible_dist[j] + feasible_dist_matrix[j,i] - feasible_dist[i]
    
    # Let's compute this properly
    # For fixed i, we want min over j != i of (feasible_dist[j] + feasible_dist_matrix[j,i] - feasible_dist[i])
    
    # feasible_dist_matrix[j,i] is the distance from node j to node i
    # So for each i, column i of feasible_dist_matrix gives dist(j, i) for all j
    
    # Let's build the regret scores
    # regret[i] = min_{j != i} (feasible_dist[j] + dist(j,i) - feasible_dist[i])
    
    # Vectorized:
    # For each i, we have an array of values for all j:
    #   val[i,j] = feasible_dist[j] + feasible_dist_matrix[j,i] - feasible_dist[i]
    
    # We can compute this as:
    #   val = feasible_dist[None, :] + feasible_dist_matrix.T - feasible_dist[:, None]
    # Then set diagonal to inf and take min along axis 1
    
    val_matrix = feasible_dist[None, :] + feasible_dist_matrix.T - feasible_dist[:, None]
    
    # Set diagonal to inf so we don't consider j=i
    np.fill_diagonal(val_matrix, np.inf)
    
    # For each i, regret[i] = min over j of val_matrix[i,j]
    # But we want: for node i, how much worse is the best alternative?
    # Actually, let me reconsider.
    
    # The regret should measure: if I don't visit i now, what's the cost?
    # The best alternative to visiting i now is to visit some j first.
    # The extra cost of visiting j before i (compared to i directly) is:
    #   dist(current, j) + dist(j, i) - dist(current, i)
    
    # If this extra cost is small for some j, then it's not costly to delay i.
    # If this extra cost is large for ALL j, then i has high regret (should be visited now).
    
    # So regret[i] = min_{j != i} (dist(current, j) + dist(j, i) - dist(current, i))
    # High regret = should visit i now
    
    # Wait, that's backwards. If the min extra cost is HIGH, it means ALL alternatives are costly,
    # so i should be visited now. If the min extra cost is LOW, some alternative is cheap,
    # so we can delay i.
    
    # So regret[i] = min_{j != i} (feasible_dist[j] + dist(j,i) - feasible_dist[i])
    # We want to select the node with HIGHEST regret.
    
    regret = np.min(val_matrix, axis=1)
    
    # Now combine regret with other factors
    # Adaptive alpha based on capacity
    avg_demand = np.mean(unvisited_demands) if n > 0 else 1.0
    capacity_ratio = rest_capacity / (avg_demand + 1e-9)
    
    # When capacity is tight, prioritize nodes that fit well (lower demand) and are close
    # When capacity is loose, prioritize regret-based selection
    
    # Weight for regret vs. distance
    # High capacity ratio -> more weight on regret
    # Low capacity ratio -> more weight on distance to depot (close route)
    
    regret_weight = 0.3 + 0.4 / (1.0 + 0.2 * capacity_ratio)
    dist_weight = 1.0 - regret_weight
    
    # Normalize regret to [0, 1] range for combining
    regret_min = np.min(regret)
    regret_max = np.max(regret)
    if regret_max - regret_min > 1e-9:
        regret_norm = (regret - regret_min) / (regret_max - regret_min)
    else:
        regret_norm = np.zeros_like(regret)
    
    # Normalize distances
    dist_max = np.max(feasible_dist)
    if dist_max > 1e-9:
        dist_norm = feasible_dist / dist_max
    else:
        dist_norm = np.zeros_like(feasible_dist)
    
    # Combined score: lower is better
    # We want high regret (good) and low distance (good)
    # So: score = dist_weight * dist_norm - regret_weight * regret_norm
    # Or: score = dist_weight * dist_norm + regret_weight * (1 - regret_norm)
    
    scores = dist_weight * dist_norm + regret_weight * (1.0 - regret_norm)
    
    # Add savings consideration: high savings means visiting this node is beneficial
    # Normalize savings
    savings_min = np.min(savings)
    savings_max = np.max(savings)
    if savings_max - savings_min > 1e-9:
        savings_norm = (savings - savings_min) / (savings_max - savings_min)
    else:
        savings_norm = np.zeros_like(savings)
    
    # Add a small weight for savings
    savings_weight = 0.15
    scores = scores + savings_weight * (1.0 - savings_norm)
    
    # Stuck penalty: if remaining capacity after visiting is very low, add penalty
    remaining_after = rest_capacity - feasible_demands
    stuck_penalty = np.where(remaining_after <= 1, 0.1, 0.0)
    scores = scores + stuck_penalty
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
