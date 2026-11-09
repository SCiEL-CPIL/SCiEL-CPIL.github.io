"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for precomputed data
_cache = {}
_deadline = None

def _get_deadline():
    global _deadline
    if _deadline is None:
        _deadline = time.time() + 110.0  # safety margin before 120s
    return _deadline

def _time_exceeded():
    return time.time() > _get_deadline()

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node to visit using a multi-criteria heuristic:
    1. Filter to feasible nodes (demand fits in remaining capacity)
    2. If no feasible nodes, return depot (0)
    3. Score each feasible node using a combination of:
       - Distance from current node (nearest neighbor)
       - Savings-based scoring (how much we save by visiting this node now)
       - Lookahead: estimated cost to return to depot
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Check time budget - if exceeded, fall back to simple nearest feasible
    if _time_exceeded():
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    # Filter to feasible nodes
    feasible = []
    for n in unvisited_nodes:
        if demands[n] <= rest_capacity:
            feasible.append(n)
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible = np.array(feasible)
    
    # Compute distances from current node to all feasible nodes
    dists_from_current = distance_matrix[current_node, feasible]
    
    # Compute distances from each feasible node back to depot
    dists_to_depot = distance_matrix[feasible, depot]
    
    # Compute distances from each feasible node to other unvisited nodes (for lookahead)
    # Use a lightweight 1-step lookahead: for each candidate, estimate the min distance
    # to any remaining unvisited node (excluding itself)
    n_unvisited = len(unvisited_nodes)
    unvisited_arr = np.array(unvisited_nodes)
    
    # For each feasible candidate, compute min distance to any unvisited node not equal to it
    # This is O(|feasible| * |unvisited|) which is manageable for 50 nodes
    lookahead_costs = np.zeros(len(feasible))
    for i, cand in enumerate(feasible):
        # Distances from candidate to all unvisited nodes
        dists_to_unvisited = distance_matrix[cand, unvisited_arr]
        # Mask out the candidate itself
        mask = unvisited_arr != cand
        if mask.any():
            lookahead_costs[i] = np.min(dists_to_unvisited[mask])
        else:
            lookahead_costs[i] = 0.0
    
    # Score: we want to minimize total route length.
    # A good heuristic is to minimize:
    #   w1 * dist_from_current + w2 * (dist_to_depot - min_dist_to_depot) + w3 * lookahead
    # But a simpler and effective approach: use savings-based logic.
    # 
    # Savings approach: visiting node i next saves us the detour.
    # Actually, let's use a weighted combination:
    # - Primary: distance from current (greedy nearest)
    # - Secondary: how much closer this node is to depot vs other options (encourages closing routes)
    # - Tertiary: lookahead cost (avoid choices that lead to expensive completions)
    
    # Normalize each component to [0, 1] range for weighting
    max_dist_cur = np.max(dists_from_current)
    if max_dist_cur > 0:
        norm_dist_cur = dists_from_current / max_dist_cur
    else:
        norm_dist_cur = np.zeros(len(feasible))
    
    max_dist_depot = np.max(dists_to_depot)
    if max_dist_depot > 0:
        norm_dist_depot = dists_to_depot / max_dist_depot
    else:
        norm_dist_depot = np.zeros(len(feasible))
    
    max_lookahead = np.max(lookahead_costs)
    if max_lookahead > 0:
        norm_lookahead = lookahead_costs / max_lookahead
    else:
        norm_lookahead = np.zeros(len(feasible))
    
    # Weighted score: lower is better
    # Emphasize nearest neighbor, but also consider depot proximity and lookahead
    w1 = 0.5   # distance from current
    w2 = 0.2   # distance to depot (prefer nodes closer to depot to close route)
    w3 = 0.3   # lookahead cost
    
    scores = w1 * norm_dist_cur + w2 * norm_dist_depot + w3 * norm_lookahead
    
    # Select the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
