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
    n_total = len(unvisited)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 0:
        return depot
    
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node and depot to all feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes].astype(np.float64)
    dist_from_depot = distance_matrix[depot, feasible_nodes].astype(np.float64)
    
    # === STRUCTURAL CHANGE: 1-step lookahead insertion cost ===
    # For each candidate node i, estimate the marginal cost of visiting i now.
    # Marginal cost = dist(current, i) + estimated_cost_after_i
    # where estimated_cost_after_i considers:
    #   - If we go to depot after i: dist(i, depot)
    #   - If we go to another feasible node j after i: dist(i, j) + dist(j, depot)
    # We take the minimum of these as the best-case continuation cost.
    # This is a 1-step lookahead that captures how visiting i now affects the rest of the route.
    
    # Compute pairwise distances among feasible nodes
    # dist_feasible[i, j] = distance between feasible_nodes[i] and feasible_nodes[j]
    dist_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)].astype(np.float64)
    
    # For each candidate i, compute the best continuation cost after visiting i
    # Option 1: Go directly to depot: dist(i, depot) = dist_from_depot[i]
    # Option 2: Go to some other feasible node j != i, then to depot:
    #   min_{j != i} [dist(i, j) + dist(j, depot)]
    # But we also need to check capacity: after visiting i, remaining cap = rest_capacity - demand[i]
    # For the lookahead, we approximate by assuming any j with demand[j] <= rest_capacity - demand[i] is feasible
    
    # Compute demand of feasible nodes
    feasible_demands = np.array([demands[j] for j in feasible_nodes], dtype=np.float64)
    
    # For each i, find min_{j != i, demand[j] <= rest_capacity - demand[i]} [dist(i,j) + dist(j, depot)]
    # This is O(n_feasible^2) which is fine for n <= 50
    
    best_continuation = np.full(n_feasible, np.inf, dtype=np.float64)
    
    for i in range(n_feasible):
        # Option 1: go to depot
        best_continuation[i] = dist_from_depot[i]
        
        # Option 2: go to another feasible node j
        remaining_after_i = rest_capacity - feasible_demands[i]
        # j must satisfy: j != i and demand[j] <= remaining_after_i
        for j in range(n_feasible):
            if j == i:
                continue
            if feasible_demands[j] <= remaining_after_i:
                cost = dist_feasible[i, j] + dist_from_depot[j]
                if cost < best_continuation[i]:
                    best_continuation[i] = cost
    
    # Total estimated cost if we visit node i next:
    total_cost_i = dist_from_current[i] + best_continuation[i]
    
    # We want to minimize total_cost_i
    # But we also need to consider that going to depot might be better if all remaining
    # nodes are far away or capacity is tight
    
    # Compute the cost of going to depot now (closing the route)
    # This is just dist(current, depot), but then we'd need new routes for remaining nodes
    # For the heuristic, we compare: is visiting any node worth it vs closing?
    # If the best total_cost_i is much larger than dist(current, depot) + some penalty,
    # we should close the route.
    
    dist_to_depot_now = distance_matrix[current_node, depot]
    
    # Threshold: if the cheapest node's total cost exceeds going to depot by too much,
    # close the route. The "too much" threshold depends on the scale of distances.
    # Use a relative threshold: if min(total_cost) > dist_to_depot_now * 1.5 + median_dist
    median_dist = np.median(dist_from_current)
    threshold = dist_to_depot_now + 0.3 * median_dist
    
    min_total_cost = np.min(total_cost)
    
    if min_total_cost > threshold:
        return depot
    
    # Pick the node with minimum total estimated cost
    best_idx = np.argmin(total_cost)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
