"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to pick the next customer or return to depot.
    Uses a savings-inspired greedy approach: prefer nodes that minimize the
    additional cost of visiting them, while respecting capacity.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Compute distances from current node to all feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Compute distances from depot to all feasible nodes (for savings calculation)
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Clarke-Wright savings heuristic:
    # The savings from visiting node i instead of going back to depot is:
    # savings(i) = dist(current, i) + dist(i, depot) - dist(current, depot)
    # We want to maximize savings, i.e., minimize (dist(current, i) + dist(i, depot) - dist(current, depot))
    # Since dist(current, depot) is constant for all candidates, we minimize:
    # cost(i) = dist(current, i) + dist(i, depot)
    
    cost = dist_from_current + dist_to_depot
    
    # Also consider a lookahead: if we visit node i, what's the best next move?
    # For simplicity, use a 2-step lookahead to break ties and improve quality.
    # For each candidate, estimate the minimum distance to any other feasible node or depot.
    
    # Build a set of feasible nodes for fast lookup
    feasible_set = set(feasible_nodes.tolist())
    
    # For each candidate, compute the best next destination (excluding itself)
    best_next_costs = np.zeros(len(feasible_nodes))
    for idx, node in enumerate(feasible_nodes):
        # Candidates for next: all feasible nodes except this one, plus depot
        other_nodes = [n for n in feasible_nodes if n != node]
        if other_nodes:
            next_costs = distance_matrix[node, other_nodes]
            min_next = np.min(next_costs)
        else:
            min_next = 0.0
        # The best next move could also be depot
        min_next = min(min_next, distance_matrix[node, depot])
        best_next_costs[idx] = min_next
    
    # Total estimated cost: immediate cost + discounted future cost
    # Use a weight of 0.5 for the lookahead to balance immediate and future costs
    total_cost = cost + 0.5 * best_next_costs
    
    # Pick the node with minimum total cost
    best_idx = np.argmin(total_cost)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
