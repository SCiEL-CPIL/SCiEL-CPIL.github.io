import numpy as np
import time

# Global time guard
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize time guard if not set
    if _start_time is None:
        _start_time = time.time()
        
    # Check deadline (150s limit, use 140s for safety)
    if time.time() - _start_time > 140:
        # If we are running out of time, just return nearest feasible or depot
        # This is a fallback, shouldn't happen for this complexity
        if not unvisited_nodes:
            return depot
        feasible_mask = demands[unvisited_nodes] <= rest_capacity
        if not np.any(feasible_mask):
            return depot
        feasible_nodes = unvisited_nodes[feasible_mask]
        # Fallback to nearest
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])

    if not unvisited_nodes:
        return depot

    # Filter feasible nodes based on capacity
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # Calculate density-weighted score
    # Score = (Distance to Current + Distance to Depot) / Demand
    # This favors nodes that are close to the current position and the depot,
    # and have high demand (efficient capacity usage).
    
    dist_to_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    node_demands = demands[feasible_nodes]
    
    # Avoid division by zero, though demands are >= 1
    scores = (dist_to_current + dist_to_depot) / node_demands
    
    # Select node with minimum score
    min_idx = np.argmin(scores)
    return int(feasible_nodes[min_idx])
