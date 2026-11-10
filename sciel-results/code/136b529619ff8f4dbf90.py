"""Capacitated VRP heuristic: savings-based with multi-candidate lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


def _lookahead_cost(curr, cap, remaining_set, depot, demands, dist_mat, depth=4):
    """
    Estimate the remaining route cost from current state using a greedy savings heuristic.
    Returns the estimated total cost from curr back through remaining nodes to depot.
    """
    cost = 0.0
    rem = set(remaining_set)
    
    for _ in range(depth):
        if time.time() > _DEADLINE:
            break
        if not rem:
            break
        
        # Find feasible nodes
        feasible = [r for r in rem if demands[r] <= cap]
        if not feasible:
            break
        
        # Compute savings for each feasible node
        d_curr_dep = dist_mat[curr, depot]
        best_next = None
        best_score = -np.inf
        
        for r in feasible:
            d_dep_r = dist_mat[depot, r]
            d_curr_r = dist_mat[curr, r]
            s = d_curr_dep + d_dep_r - d_curr_r
            # Blend savings with distance: prefer high savings but penalize far nodes
            score = s - 0.3 * d_curr_r
            if score > best_score:
                best_score = score
                best_next = r
        
        if best_next is None:
            break
        
        cost += dist_mat[curr, best_next]
        curr = best_next
        cap -= demands[curr]
        rem.discard(curr)
    
    # Return to depot
    if rem:
        cost += dist_mat[curr, depot]
    else:
        cost += dist_mat[curr, depot]
    
    return cost


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a multi-candidate savings heuristic.
    Generates several candidate nodes and evaluates them using a lookahead score.
    Returns 0 (depot) if no feasible node should be visited.
    """
    if time.time() > _DEADLINE:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)

    # 1. Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    # 2. If at depot, pick the nearest feasible node (standard)
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])

    # 3. Generate candidate scores
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    savings = d_current_depot + d_depot_j - d_current_j
    
    # If the best saving is negative, it's cheaper to go back to depot
    if np.max(savings) < 0:
        return depot

    # Normalization for weighting
    max_saving = np.max(savings)
    min_saving = np.min(savings)
    if max_saving > min_saving:
        norm_savings = (savings - min_saving) / (max_saving - min_saving)
    else:
        norm_savings = np.ones_like(savings)

    max_dist = np.max(d_current_j)
    min_dist = np.min(d_current_j)
    if max_dist > min_dist:
        norm_dist = (d_current_j - min_dist) / (max_dist - min_dist)
    else:
        norm_dist = np.zeros_like(d_current_j)

    # Demands
    d_j = demands[feasible_nodes].astype(float)
    max_d = np.max(d_j)
    min_d = np.min(d_j)
    if max_d > min_d:
        norm_d = (d_j - min_d) / (max_d - min_d)
    else:
        norm_d = np.zeros_like(d_j)

    # Generate diverse candidates using different scoring strategies
    candidates = []
    
    # Candidate 1: Max Savings
    idx1 = np.argmax(savings)
    candidates.append(idx1)
    
    # Candidate 2: Savings - Distance
    score2 = norm_savings - norm_dist
    idx2 = np.argmax(score2)
    candidates.append(idx2)
    
    # Candidate 3: Savings - 0.5 * Distance
    score3 = norm_savings - 0.5 * norm_dist
    idx3 = np.argmax(score3)
    candidates.append(idx3)
    
    # Candidate 4: Nearest Neighbor
    idx4 = np.argmin(d_current_j)
    candidates.append(idx4)
    
    # Candidate 5: Savings - Distance - small demand penalty
    score5 = norm_savings - norm_dist - 0.1 * norm_d
    idx5 = np.argmax(score5)
    candidates.append(idx5)
    
    # Candidate 6: Pure savings with slight distance discount
    score6 = savings - 0.2 * d_current_j
    idx6 = np.argmax(score6)
    candidates.append(idx6)
    
    # Candidate 7: Balanced - emphasize both savings and proximity
    score7 = 0.6 * norm_savings + 0.4 * (1 - norm_dist)
    idx7 = np.argmax(score7)
    candidates.append(idx7)
    
    # Remove duplicates
    unique_candidates = list(dict.fromkeys(candidates))
    
    if len(unique_candidates) == 0:
        return depot

    # Lookahead: Simulate the route for each candidate
    best_score = float('inf')
    best_node = int(feasible_nodes[unique_candidates[0]])
    
    for idx in unique_candidates:
        node = feasible_nodes[idx]
        
        # State after picking this node
        curr = node
        cap = rest_capacity - demands[node]
        remaining = set(unvisited_nodes)
        remaining.discard(node)
        
        route_cost = distance_matrix[current_node, node]
        
        # Use lookahead to estimate remaining cost
        remaining_cost = _lookahead_cost(curr, cap, remaining, depot, demands, distance_matrix, depth=4)
        route_cost += remaining_cost
        
        # We want to minimize this estimated cost
        if route_cost < best_score:
            best_score = route_cost
            best_node = node
    
    return best_node
# EVOLVE-BLOCK-END
