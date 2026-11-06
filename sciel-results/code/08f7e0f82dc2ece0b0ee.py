"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time to avoid timeout
_START_TIME = None
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _check_deadline():
    """Return True if we should stop and return depot for safety."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a combined savings + return-cost + capacity-efficiency heuristic
    with a regret-based component.
    
    Returns 0 (depot) to close route if no feasible customers or deadline exceeded.
    """
    # Deadline guard
    if _check_deadline():
        return 0
    
    if len(unvisited_nodes) == 0:
        return 0
    
    unvisited_arr = np.array(unvisited_nodes)
    
    # Filter to only feasible customers (demand <= rest_capacity)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # Get distances from current node to feasible customers
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Get distances from feasible customers back to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Get distances from depot to feasible customers (for savings calculation)
    dist_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Demands of feasible nodes
    demands_feasible = demands[feasible_nodes]
    
    # --- Combined Scoring ---
    # We want to MINIMIZE the total score.
    
    # 1. Savings component:
    #    score_savings = dist_from_current - dist_from_depot
    #    This rewards nodes that are far from depot but close to current (high savings).
    
    # 2. Return cost component:
    #    Penalize nodes that are far from the depot.
    #    score_return = lambda_return * dist_to_depot
    
    # 3. Capacity efficiency component:
    #    Penalize low cap_ratio when distance is high
    
    # 4. Regret-based component:
    #    For each candidate, compute the second-best distance from current node.
    #    Regret = second_best_dist - best_dist. Higher regret means we lose more
    #    if we don't pick this node now.
    
    # Base scores
    score_savings = dist_from_current - dist_from_depot
    score_return = 0.5 * dist_to_depot
    
    # Normalize distances for capacity penalty scaling
    avg_dist = np.mean(dist_from_current)
    if avg_dist > 0:
        dist_norm = dist_from_current / (avg_dist * 2 + 1e-9)
        cap_ratio = demands_feasible / float(rest_capacity)
        # Penalize low cap_ratio when distance is high
        cap_penalty = 0.1 * (1 - cap_ratio) * dist_norm
    else:
        cap_penalty = np.zeros_like(dist_from_current)
    
    # Regret component: 
    # Sort distances and compute regret as difference between 2nd best and best
    sorted_dists = np.sort(dist_from_current)
    if len(sorted_dists) >= 2:
        regret = sorted_dists[1] - sorted_dists[0]
        # Scale regret to be comparable to other terms
        regret_score = -0.3 * regret * np.ones_like(dist_from_current)
        # But we want to reward high regret for nodes that are close
        # So subtract regret from score for nodes that have high regret
        # Actually, regret is a property of the node - if a node has high regret,
        # we should prefer it. So we subtract a scaled regret.
        # But regret is the same for all nodes in this formulation... 
        # Let me reconsider: regret for node i is d(i, 2nd_best) - d(i, best)
        # where best and 2nd_best are the two closest nodes to i.
        # This requires pairwise computation.
        
        # Simpler regret: for each node, how much worse is the 2nd closest node
        # to the current position compared to this node?
        # This is actually: for node i, regret_i = min_{j != i} d(current, j) - d(current, i)
        # which is negative or zero. We want to reward nodes with small (negative) regret.
        # So score_regret = d(current, i) - min_{j != i} d(current, j)
        # This is >= 0, and we want to minimize it.
        
        # Compute min distance excluding self
        min_dist_excl = np.full_like(dist_from_current, np.inf)
        for idx in range(len(feasible_nodes)):
            other_dists = np.delete(dist_from_current, idx)
            if len(other_dists) > 0:
                min_dist_excl[idx] = np.min(other_dists)
        
        # Regret score: how much farther is this node than the closest other node
        score_regret = dist_from_current - min_dist_excl
    else:
        score_regret = np.zeros_like(dist_from_current)
    
    # Combined score: lower is better
    scores = score_savings + score_return + cap_penalty + score_regret
    
    # Pick the customer with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
