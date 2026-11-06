"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0


def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE


def _greedy_lookahead(current_node, depot, remaining_nodes, rest_capacity, demands, distance_matrix, max_depth=8):
    """
    Simulate the rest of the route using a greedy nearest-feasible strategy
    and return the total distance of that simulated route (excluding return to depot).
    """
    if len(remaining_nodes) == 0:
        return 0.0
    
    remaining_set = set(remaining_nodes)
    remaining_demands = demands[remaining_nodes].copy()
    total_dist = 0.0
    current = current_node
    remaining_cap = rest_capacity
    steps = 0
    
    # We simulate up to max_depth or until no feasible nodes remain
    while remaining_set and steps < max_depth:
        steps += 1
        
        # Find feasible nodes
        feasible = []
        for node in remaining_set:
            if demands[node] <= remaining_cap:
                feasible.append(node)
        
        if not feasible:
            break
        
        # Pick nearest feasible
        best_node = None
        best_dist = float('inf')
        for node in feasible:
            d = distance_matrix[current, node]
            if d < best_dist:
                best_dist = d
                best_node = node
        
        if best_node is None:
            break
        
        total_dist += best_dist
        current = best_node
        remaining_cap -= demands[best_node]
        remaining_set.remove(best_node)
        
        # Early exit if time is tight
        if _time_exceeded():
            break
    
    # If there are still remaining nodes, we can't finish them in this route,
    # but for the purpose of scoring the "current" route, we just stop.
    # However, the "cost" of leaving nodes unvisited is implicit in the fact
    # that they will start a new route from depot, which is a constant penalty
    # relative to other candidates that also leave them. But actually, the
    # choice of which nodes to visit in THIS route matters because it determines
    # the remaining set for future routes.
    
    # A better lookahead would estimate the cost of the remaining nodes in future routes,
    # but that's complex. For now, the length of the current route is the main factor.
    # Adding the return to depot is important.
    total_dist += distance_matrix[current, depot]
    
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes
    unvisited_demands = demands[unvisited]
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    
    if _time_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # 1. Immediate distance from current
    dists_current = distance_matrix[current_node, feasible_nodes]
    
    # 2. Distance to depot (return cost proxy)
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # 3. 2-step lookahead: minimum distance from each feasible node to any other feasible node
    feas_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    np.fill_diagonal(feas_dist, np.inf)
    min_next_dist = np.min(feas_dist, axis=1)
    
    # 4. Detour cost (angular consistency proxy)
    nearest_neighbor_idx = np.argmin(feas_dist, axis=1)
    nearest_neighbor_nodes = feasible_nodes[nearest_neighbor_idx]
    
    d_current_i = dists_current
    d_i_j = min_next_dist
    d_current_j = distance_matrix[current_node, nearest_neighbor_nodes]
    
    detour_cost = d_current_i + d_i_j - d_current_j
    
    # 5. Cluster centrality
    dists_feas_to_unvisited = distance_matrix[np.ix_(feasible_nodes, unvisited)]
    sum_dists = np.sum(dists_feas_to_unvisited, axis=1)
    n_unvisited = len(unvisited)
    if n_unvisited > 1:
        mean_dists = sum_dists / (n_unvisited - 1)
    else:
        mean_dists = np.zeros(n_feasible)
    
    # Base heuristic score
    base_scores = (1.0 * dists_current + 
                   0.3 * dists_depot + 
                   0.5 * detour_cost + 
                   0.2 * min_next_dist + 
                   0.1 * mean_dists)
    
    # If we have enough time, use full greedy lookahead for top candidates
    # to break ties or refine the choice.
    # To save time, we only do this if the number of feasible nodes is not too large
    # or if we are early in the process.
    
    # Heuristic: if n_feasible is small, we can afford lookahead.
    # Also, if we are not close to deadline.
    
    # Let's try lookahead on the top 3 candidates from base_scores
    top_k = min(3, n_feasible)
    top_indices = np.argsort(base_scores)[:top_k]
    
    lookahead_scores = np.zeros(top_k)
    for k, idx in enumerate(top_indices):
        candidate_node = feasible_nodes[idx]
        
        # Simulate route starting from candidate_node
        # Remaining nodes are all unvisited nodes except candidate_node
        # But we must respect capacity.
        
        # Create a list of remaining nodes for simulation
        # We pass the full unvisited set, but the simulation will handle capacity.
        # Actually, the simulation function needs to know the current node is candidate_node.
        # And the remaining capacity is rest_capacity - demands[candidate_node].
        
        rem_cap = rest_capacity - demands[candidate_node]
        
        # The set of nodes to consider for the rest of the route is unvisited_nodes minus candidate_node
        # But we should only consider nodes that are feasible at each step.
        
        # To speed up simulation, we can just pass the list of unvisited nodes
        # and let the simulation filter them.
        
        # Optimization: if rem_cap is 0, no more nodes can be visited.
        if rem_cap <= 0:
            lookahead_scores[k] = distance_matrix[current_node, candidate_node] + distance_matrix[candidate_node, depot]
        else:
            # We need to simulate.
            # To avoid modifying the original arrays, we pass copies or handle it carefully.
            # The _greedy_lookahead function takes lists/arrays.
            
            # We pass the unvisited nodes (including candidate, but it will be removed in first step? No, current is candidate).
            # Wait, my _greedy_lookahead assumes current_node is the start.
            # So we call it with current_node=candidate_node, and remaining_nodes = unvisited_nodes (it will filter feasible).
            # But candidate_node is in unvisited_nodes. The function removes nodes it visits.
            # It starts by finding feasible nodes from remaining_set.
            # candidate_node is NOT in remaining_set if we pass unvisited_nodes without candidate?
            # No, the route is: Depot -> ... -> current_node -> candidate_node -> ... -> Depot.
            # So the "remaining" nodes for the simulation are the unvisited nodes EXCLUDING candidate_node.
            
            rem_nodes_list = [n for n in unvisited_nodes if n != candidate_node]
            
            lookahead_scores[k] = _greedy_lookahead(candidate_node, depot, rem_nodes_list, rem_cap, demands, distance_matrix, max_depth=10)
        
        if _time_exceeded():
            break
    
    # Combine base score and lookahead score
    # The base score is in units of distance (roughly).
    # The lookahead score is total route length.
    # We want to minimize total route length.
    # A simple way is to add them, but they are on different scales.
    # Base score is a heuristic for the *next step* quality.
    # Lookahead is the *actual* cost of the resulting route (approx).
    
    # Let's weight the lookahead heavily if it was computed.
    # If lookahead was not computed for all (due to time), we fall back to base.
    
    # For simplicity, let's just pick the candidate with the lowest lookahead score among the top_k.
    # If top_k < n_feasible, we assume the top_k are good enough.
    
    best_local_idx = np.argmin(lookahead_scores)
    best_global_idx = top_indices[best_local_idx]
    
    return int(feasible_nodes[best_global_idx])
# EVOLVE-BLOCK-END
