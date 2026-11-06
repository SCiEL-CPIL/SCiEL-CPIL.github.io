"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time once at module load
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 2.0  # return safe result this far before deadline

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Multi-Objective Adaptive Scoring.
    
    Strategy:
    1. Filter unvisited nodes by capacity.
    2. If none fit, return depot.
    3. If only one fits, return it.
    4. Compute a composite score for each feasible node:
       - Proximity: Distance from current node to candidate.
       - Savings: Benefit of visiting this node now vs. starting a new route or visiting later.
         Approximated by (dist(depot, current) + dist(depot, cand) - dist(current, cand)).
         We want to MAXIMIZE savings, so we subtract it from the score.
       - Urgency: If rest_capacity is low or few nodes remain, prioritize nodes that allow
         returning to the depot efficiently (small dist(cand, depot)).
    
    The weights of these components adapt based on the state:
    - 'phase': early (many nodes, high cap) vs late (few nodes, low cap).
    - In early phase, prioritize savings and proximity to build efficient clusters.
    - In late phase, prioritize closing the route (return to depot).
    """
    if len(unvisited_nodes) == 0:
        return depot

    # If time is running out, fall back to simple nearest feasible
    if _time_exceeded():
        feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
        feasible_nodes = unvisited_nodes[feasible_mask]
        if len(feasible_nodes) == 0:
            return depot
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])

    # Filter by capacity
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Distances
    d_cur_cand = distance_matrix[current_node, feasible_nodes]
    d_depot_cur = distance_matrix[depot, current_node]
    d_depot_cand = distance_matrix[depot, feasible_nodes]
    
    # Calculate Savings:
    # Savings of adding cand to the route ending at cur, compared to serving them separately?
    # Actually, in a construction heuristic, the "savings" of linking i and j is d(0,i)+d(0,j)-d(i,j).
    # Here we are extending a path. The cost added is d(cur, cand).
    # If we don't visit cand now, we might have to start a new route or visit it later.
    # A proxy for "efficiency" is how much we save by visiting cand now rather than going back to depot and then to cand?
    # Cost if we go back to depot then to cand: d(cur, 0) + d(0, cand).
    # Cost if we go directly: d(cur, cand).
    # Saving = d(cur, 0) + d(0, cand) - d(cur, cand).
    # We want to maximize this saving. So we subtract it from the cost score.
    
    savings = d_depot_cur + d_depot_cand - d_cur_cand
    
    # Determine phase
    # If rest_capacity is small relative to total demand or if few nodes left, we are in "closing" phase.
    # Let's use the number of unvisited nodes as a proxy for phase.
    n_unvisited = len(unvisited_nodes)
    
    # Weights
    # w_prox: weight for immediate distance (lower is better)
    # w_save: weight for savings (higher is better, so we subtract w_save * savings)
    # w_ret: weight for distance to depot (lower is better, helps closing)
    
    if n_unvisited > 15:
        # Early phase: build efficient routes. High savings, moderate proximity.
        w_prox = 1.0
        w_save = 0.8
        w_ret = 0.1
    elif n_unvisited > 5:
        # Mid phase: balance.
        w_prox = 1.2
        w_save = 0.5
        w_ret = 0.3
    else:
        # Late phase: close routes. High weight on returning to depot.
        w_prox = 1.0
        w_save = 0.2
        w_ret = 1.5

    # Composite Score
    # Minimize: w_prox * d(cur, cand) - w_save * savings + w_ret * d(cand, depot)
    # Note: savings can be negative if triangle inequality is loose or geometry is weird, but usually positive.
    # If savings is negative, it means going via depot is "cheaper" than direct? No, triangle inequality says d(cur,cand) <= d(cur,0)+d(0,cand), so savings >= 0.
    
    scores = w_prox * d_cur_cand - w_save * savings + w_ret * d_depot_cand
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
