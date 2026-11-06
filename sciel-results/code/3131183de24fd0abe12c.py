"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # Score each feasible node: balance distance from current with
    # the "savings" of not going back to depot yet.
    # cost = dist(current, node) - alpha * (dist(current, depot) + dist(node, depot) - 2*dist(node, depot))
    # Simplified: prefer nodes that are close AND where visiting them saves vs going to depot
    d_current_to_nodes = distance_matrix[current_node, feasible_nodes]
    d_nodes_to_depot = distance_matrix[feasible_nodes, depot]
    d_current_to_depot = distance_matrix[current_node, depot]
    
    # Clarke-Wright style savings: visiting node saves (d_current_depot + d_node_depot - d_current_node)
    # We want to maximize savings / minimize effective cost
    # effective_cost = d_current_node - beta * savings
    # where savings = d_current_depot + d_node_depot - d_current_node
    # So effective_cost = d_current_node - beta*(d_current_depot + d_node_depot - d_current_node)
    #                   = d_current_node*(1+beta) - beta*(d_current_depot + d_node_depot)
    beta = 0.5
    effective_cost = d_current_to_nodes * (1.0 + beta) - beta * (d_current_to_depot + d_nodes_to_depot)
    
    # Also consider: if rest_capacity after visiting is very small, prefer going to depot
    # to avoid being stranded
    remaining_after = rest_capacity - demands[feasible_nodes]
    # Penalty if remaining capacity is too small to visit any other node
    min_demand = np.min(demands)
    if np.min(remaining_after) < min_demand:
        # Strongly prefer depot if we can't visit any more nodes
        effective_cost[remaining_after < min_demand] += 1000.0
    
    best_idx = int(np.argmin(effective_cost))
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
