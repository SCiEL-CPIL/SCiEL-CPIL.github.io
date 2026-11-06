"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return depot if we've exceeded 145s
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)
    
    if n == 0:
        return depot
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited]
    
    # Get demands of unvisited nodes
    unvisited_demands = demands[unvisited]
    
    # Feasibility: demand must fit in remaining capacity
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    feasible_dist = dist_from_current[feasible_indices]
    feasible_demands = unvisited_demands[feasible_indices]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Compute distance from each feasible candidate to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Adaptive alpha: weight for distance to depot
    # When rest_capacity is low relative to average demand, close the route sooner (higher alpha)
    # When rest_capacity is high, prefer nearer customers (lower alpha)
    avg_demand = np.mean(unvisited_demands) if n > 0 else 1.0
    capacity_ratio = rest_capacity / (avg_demand + 1e-9)
    
    # Map capacity_ratio to alpha: 
    # capacity_ratio ~ 1 -> alpha ~ 0.6 (close route soon)
    # capacity_ratio ~ 5+ -> alpha ~ 0.15 (keep route going)
    # Use a smooth mapping
    alpha = 0.15 + 0.45 / (1.0 + 0.3 * capacity_ratio)
    
    # Base score: distance from current + alpha * distance to depot
    scores = feasible_dist + alpha * dist_to_depot
    
    # Additional refinement: penalize candidates that would leave very little capacity
    # but not enough to be infeasible, to avoid getting stuck
    remaining_after = rest_capacity - feasible_demands
    # Soft penalty: if remaining_after is very small, add a small penalty
    # This is a smooth surrogate for the "will I get stuck?" constraint
    stuck_penalty = np.where(remaining_after <= 1, 0.1 * feasible_dist, 0.0)
    scores = scores + stuck_penalty
    
    # Use a soft-min inspired approach: rank-based scoring to smooth the decision
    # Compute rank of each score (lower rank = better)
    # Apply a mild nonlinearity to emphasize the best candidate
    sorted_scores = np.sort(scores)
    # Soft-min: use log-sum-exp to get a smooth minimum estimate
    # Then score each candidate by how close it is to the soft-min
    # For discrete selection, this effectively does argmin but with tie-breaking by proximity to soft-min
    
    # Simpler: just pick the argmin, but use a temperature to break near-ties
    # by slightly favoring candidates with lower dist_to_depot among near-min scores
    min_score = np.min(scores)
    threshold = min_score + 0.01 * min_score  # 1% tolerance
    
    # Candidates within 1% of the best score
    close_mask = scores <= threshold
    if np.sum(close_mask) > 1:
        # Among close candidates, prefer the one with lower dist_to_depot
        close_indices = np.where(close_mask)[0]
        close_depot_dists = dist_to_depot[close_indices]
        best_close = close_indices[np.argmin(close_depot_dists)]
        return int(feasible_nodes[best_close])
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
