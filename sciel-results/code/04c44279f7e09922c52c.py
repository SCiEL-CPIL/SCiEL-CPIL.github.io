"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We will implement a heuristic that uses a 2-step lookahead with a savings-like penalty.
# The key insight from the measured knowledge is that 1-step lookahead improved over pure greedy.
# We'll go further: for each candidate next node, we estimate the cost of going to that node AND
# then to the best feasible next-next node (or depot if no feasible next-next).
# We also incorporate a "savings" component: prefer nodes that create large savings relative to
# serving them individually.

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0) to close a route.
    
    Uses a 2-step lookahead heuristic with savings-based scoring.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Deadline guard: if we've been running too long, fall back to nearest feasible
    # (This is a per-call guard; the real deadline is managed by the evaluator)
    
    n = len(unvisited_nodes)
    if n == 1:
        # Only one node left: check if it fits
        if demands[unvisited_nodes[0]] <= rest_capacity:
            return int(unvisited_nodes[0])
        else:
            return depot
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Filter to feasible nodes (demand <= rest_capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return depot
    
    feasible_nodes = unvisited_nodes[feasible_indices]
    feasible_dist = dist_from_current[feasible_indices]
    
    # For each feasible candidate, compute a 2-step lookahead score
    # Score = dist(current->candidate) + estimated best next step from candidate
    # We estimate the next step as the minimum of:
    #   - distance to nearest feasible node (with remaining capacity after candidate)
    #   - distance to depot (if we'd close the route)
    
    best_score = float('inf')
    best_node = int(feasible_nodes[0])
    
    # Precompute: for each feasible candidate, what's the best 2nd step?
    for i in range(len(feasible_indices)):
        cand = feasible_nodes[i]
        cost_to_cand = feasible_dist[i]
        
        # Remaining capacity after visiting cand
        remaining_cap = rest_capacity - demands[cand]
        
        if remaining_cap <= 0:
            # Must return to depot after this
            total_cost = cost_to_cand + distance_matrix[cand, depot]
        else:
            # Find best next step from cand among remaining feasible nodes
            # Exclude cand itself
            other_mask = np.ones(n, dtype=bool)
            other_mask[feasible_indices[i]] = False  # exclude self
            other_feasible_mask = other_mask & feasible_mask
            
            if np.any(other_feasible_mask):
                other_nodes = unvisited_nodes[other_feasible_mask]
                # Filter to those that fit in remaining capacity
                other_fit_mask = demands[other_nodes] <= remaining_cap
                if np.any(other_fit_mask):
                    other_fit_nodes = other_nodes[other_fit_mask]
                    dist_to_others = distance_matrix[cand, other_fit_nodes]
                    best_next = np.min(dist_to_others)
                    # We don't add the return to depot yet; just estimate the 2nd leg
                    total_cost = cost_to_cand + best_next
                else:
                    # No other node fits; must return to depot
                    total_cost = cost_to_cand + distance_matrix[cand, depot]
            else:
                # No other feasible nodes at all
                total_cost = cost_to_cand + distance_matrix[cand, depot]
        
        # Savings bonus: prefer nodes that are "far" from depot (high individual service cost)
        # This encourages clustering
        savings_bonus = distance_matrix[depot, cand]
        
        # Combined score: minimize total_cost, but also consider savings
        # Use a weighted combination
        score = total_cost - 0.3 * savings_bonus
        
        if score < best_score:
            best_score = score
            best_node = int(cand)
    
    return best_node
# EVOLVE-BLOCK-END
