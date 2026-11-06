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
    Select the next node using a combined savings + return-cost + capacity-efficiency heuristic.
    
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
    
    # 1. Savings component (from current program):
    #    Original savings = dist(depot, i) + dist(depot, j) - dist(i, j)
    #    Since dist(depot, i) is constant, maximizing savings is equivalent to minimizing:
    #    - (dist_from_depot - dist_from_current) = dist_from_current - dist_from_depot
    #    So we use: score_savings = dist_from_current - dist_from_depot
    #    This rewards nodes that are far from depot but close to current (high savings).
    
    # 2. Return cost component (from inspiration 2):
    #    Penalize nodes that are far from the depot, as visiting them might force
    #    an early return or make the final leg long.
    #    score_return = lambda_return * dist_to_depot
    
    # 3. Capacity efficiency component (from inspiration 2):
    #    Prefer nodes with higher demand relative to remaining capacity if they are far,
    #    to avoid wasting capacity on small, distant deliveries.
    #    But this should be a subtle adjustment.
    
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
    
    # Combined score: lower is better
    # The savings term is dominant. We subtract it (or add the negative) to minimize.
    # Actually, let's just sum them.
    # score_savings is negative when savings are high. So minimizing it is correct.
    scores = score_savings + score_return + cap_penalty
    
    # Pick the customer with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
