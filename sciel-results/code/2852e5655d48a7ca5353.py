"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # Filter feasible nodes
    feasible_mask = np.array([demands[n] <= rest_capacity for n in unvisited_nodes])
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # Distances
    dists_current = distance_matrix[current_node, feasible_nodes]
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # Heuristic Selection
    if current_node == depot:
        # Start with nearest neighbor to establish a direction
        best_idx = np.argmin(dists_current)
    else:
        # Combined savings + regret heuristic
        # Savings-based: minimize (dist to current - alpha * dist to depot)
        # This favors nodes that are close to current but far from depot,
        # encouraging long, efficient routes.
        # Regret: prefer nodes with high demand (to reduce number of trips)
        # and nodes that are far from depot (harder to serve later)
        
        alpha = 0.85
        scores = dists_current - alpha * dists_depot
        
        # Add a small demand-based urgency component
        # Higher demand nodes should be prioritized to reduce total trips
        demand_arr = np.array([demands[n] for n in feasible_nodes], dtype=float)
        max_demand = demand_arr.max() if len(demand_arr) > 0 else 1.0
        if max_demand > 0:
            demand_urgency = demand_arr / max_demand
            # Subtract a small multiple of demand urgency to prioritize high-demand nodes
            scores = scores - 0.15 * demand_urgency
        
        # Soft-min via log-sum-exp for smoother scoring
        # This provides a differentiable approximation to the min operation
        # and can help avoid local minima in the greedy selection
        
        # Compute a soft-min score: lower is better
        # Use temperature parameter to control smoothness
        temp = 2.0
        # Scale scores to be in a reasonable range for log-sum-exp
        scores_scaled = scores / temp
        # Clip to avoid overflow
        scores_scaled = np.clip(scores_scaled, -50, 50)
        
        # Log-sum-exp for soft-min: -logsumexp(-x) approximates min(x)
        # But we want to select based on the original scores, so we use
        # the original scores for argmin but could use soft-min for weighting
        # For now, keep the argmin approach but with adjusted parameters
        
        best_idx = np.argmin(scores)
        
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
