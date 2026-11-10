"""Capacitated VRP heuristic: savings-based with multi-candidate lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


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
    # We want to balance "savings" (cluster efficiency) with "distance" (proximity)
    # and potentially "demand" (urgency).
    
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

    # Strategy: Try a few different weight combinations for the score.
    # score = w1 * norm_savings - w2 * norm_dist + w3 * (1 - norm_d)
    # High savings is good. High dist is bad (penalty). High demand is good (urgency)? 
    # Actually, serving high demand early is often better to avoid stranding small demands later?
    # Or is it better to serve small demands to fill gaps?
    # Let's try:
    # 1. Pure savings (current)
    # 2. Savings - Distance
    # 3. Savings - Distance + Demand
    
    candidates = []
    
    # Candidate 1: Max Savings
    idx1 = np.argmax(savings)
    candidates.append(idx1)
    
    # Candidate 2: Savings normalized - Distance normalized
    score2 = norm_savings - norm_dist
    idx2 = np.argmax(score2)
    candidates.append(idx2)
    
    # Candidate 3: Savings normalized - 0.5 * Distance normalized
    score3 = norm_savings - 0.5 * norm_dist
    idx3 = np.argmax(score3)
    candidates.append(idx3)
    
    # Candidate 4: Nearest Neighbor
    idx4 = np.argmin(d_current_j)
    candidates.append(idx4)
    
    # Candidate 5: Savings - Distance - 0.1 * Demand (prefer low demand if close?)
    score5 = norm_savings - norm_dist - 0.1 * norm_d
    idx5 = np.argmax(score5)
    candidates.append(idx5)
    
    # Remove duplicates
    unique_candidates = list(dict.fromkeys(candidates))
    
    if len(unique_candidates) == 0:
        return depot

    # Lookahead: Simulate the route for each candidate for a few steps
    # and estimate the total cost.
    
    best_score = float('inf')
    best_node = int(feasible_nodes[unique_candidates[0]])
    
    # Shallow copy of state for simulation
    # We only simulate 3-4 steps deep to save time
    
    for idx in unique_candidates:
        node = feasible_nodes[idx]
        
        # State after picking this node
        curr = node
        cap = rest_capacity - demands[node]
        remaining = set(unvisited_nodes)
        remaining.remove(node)
        
        route_cost = distance_matrix[current_node, node]
        
        # Greedy completion for a few steps
        steps = 3
        for _ in range(steps):
            if time.time() > _DEADLINE:
                break
            if not remaining:
                break
            
            # Find feasible neighbors
            # Optimization: only consider nodes in remaining that fit in cap
            feasible_rem = []
            for r in remaining:
                if demands[r] <= cap:
                    feasible_rem.append(r)
            
            if not feasible_rem:
                break
            
            # Pick the next node using the same heuristic (simplified)
            # Use savings + distance heuristic
            d_curr_dep = distance_matrix[curr, depot]
            best_next = None
            best_s = -np.inf
            
            # For speed, we can just pick the one with max savings among feasible_rem
            # Vectorize this inner loop? It's small (<=50), so Python loop is okay if steps are small
            for r in feasible_rem:
                d_dep_r = distance_matrix[depot, r]
                d_curr_r = distance_matrix[curr, r]
                s = d_curr_dep + d_dep_r - d_curr_r
                if s > best_s:
                    best_s = s
                    best_next = r
            
            if best_next is None:
                break
            
            route_cost += distance_matrix[curr, best_next]
            curr = best_next
            cap -= demands[curr]
            remaining.remove(curr)
            
            if cap < 0:
                break
        
        # Cost to return to depot
        if remaining:
             # If we stopped early, we might need to go back to depot then continue?
             # For a rough estimate, just go back to depot.
             route_cost += distance_matrix[curr, depot]
        else:
             route_cost += distance_matrix[curr, depot]
        
        # We want to minimize this estimated cost
        if route_cost < best_score:
            best_score = route_cost
            best_node = node
    
    return best_node
# EVOLVE-BLOCK-END
