"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed data to speed up repeated calls
_cache = {}
_start_time = time.time()
_DEADLINE = 120.0  # safety margin before 150s kill
_SOFT_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for Capacitated VRP:
    Multi-factor scoring with 2-step lookahead and capacity-aware weighting.
    Returns the best customer or depot (0) if no feasible customer.
    """
    if len(unvisited_nodes) == 0:
        return 0

    # Safety deadline guard
    if _time_left() < 0.5:
        return 0

    # Check if any feasible customers exist
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    # Get feasible unvisited nodes
    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feas = len(feasible_nodes)

    # Compute distances from current node to feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Compute return-to-depot distances from feasible nodes
    return_dists = distance_matrix[feasible_nodes, depot]
    
    # Compute demands of feasible nodes
    feas_demands = demands[feasible_nodes]

    # Capacity utilization: how much of remaining capacity we'd use
    cap_util = feas_demands / max(rest_capacity, 1.0)

    # 2-step lookahead: for each candidate, estimate the cost of the next best move
    # After visiting candidate i, remaining capacity = rest_capacity - demand_i
    # We estimate the cheapest next step from candidate i to any other feasible node
    # or back to depot, weighted by feasibility
    
    lookahead_costs = np.zeros(n_feas)
    for i in range(n_feas):
        node_i = feasible_nodes[i]
        rem_cap_i = rest_capacity - feas_demands[i]
        if rem_cap_i <= 0:
            # Must return to depot after this node
            lookahead_costs[i] = return_dists[i]
            continue
        # Find best next node from node_i among feasible nodes (excluding node_i)
        # and consider return to depot
        # Use a simplified estimate: min distance from node_i to any other feasible node
        # that still fits in remaining capacity, plus its return distance
        other_mask = np.ones(n_feas, dtype=bool)
        other_mask[i] = False
        other_feasible = feasible_nodes[other_mask]
        other_demands = feas_demands[other_mask]
        
        # Which other nodes fit in remaining capacity?
        fits_mask = other_demands <= rem_cap_i
        if np.any(fits_mask):
            fitting_nodes = other_feasible[fits_mask]
            # Cost to go to a fitting node and then return to depot
            costs_to_fitting = distance_matrix[node_i, fitting_nodes] + distance_matrix[fitting_nodes, depot]
            best_next = np.min(costs_to_fitting)
        else:
            best_next = return_dists[i]
        
        # Also consider just returning to depot now
        # Take the min: either go to best next node or return to depot
        lookahead_costs[i] = min(best_next, return_dists[i])

    # Adaptive weights based on capacity situation
    # When capacity is tight, prefer lower-demand nodes more strongly
    cap_pressure = 1.0 - (rest_capacity / 50.0)  # 0 when full, 1 when nearly empty
    cap_pressure = np.clip(cap_pressure, 0.0, 1.0)

    w_dist = 1.0
    w_return = 0.2 + 0.1 * cap_pressure
    w_demand = 0.3 + 0.4 * cap_pressure
    w_lookahead = 0.3

    # Score: lower is better
    scores = (w_dist * dists + 
              w_return * return_dists + 
              w_demand * feas_demands + 
              w_lookahead * lookahead_costs)

    # Add a small penalty for nodes that are very far from depot (to avoid stragglers)
    # This encourages clustering
    depot_dist_penalty = 0.05 * return_dists
    scores += depot_dist_penalty

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
