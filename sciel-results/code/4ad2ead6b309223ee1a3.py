"""Improved distance-primary angular sweep heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a distance-primary angular sweep heuristic.
    
    This heuristic prioritizes minimizing the travel distance from the current node,
    while penalizing large angular deviations at the depot to maintain a consistent
    sweep direction and reduce backtracking. Using (1 - cos(theta)) as the angular
    penalty provides a smooth gradient that strongly discourages backtracking.
    """
    depot = int(depot)
    current_node = int(current_node)
    
    # Filter unvisited nodes by capacity feasibility
    feasible = []
    for node in unvisited_nodes:
        node = int(node)
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if not feasible:
        return 0  # Close route, return to depot
    
    # Pre-compute distance from depot to current node
    d_depot_current = distance_matrix[depot, current_node]
    
    best_node = feasible[0]
    best_score = float('inf')
    
    for node in feasible:
        node = int(node)
        d_depot_node = distance_matrix[depot, node]
        d_current_node = distance_matrix[current_node, node]
        
        # Compute cosine of the angle at the depot
        a = d_depot_current
        b = d_depot_node
        c = d_current_node
        
        if a < 1e-12 or b < 1e-12:
            cos_theta = 1.0  # Assume aligned if degenerate
        else:
            cos_theta = (a*a + b*b - c*c) / (2.0 * a * b)
            cos_theta = max(-1.0, min(1.0, cos_theta))
        
        # Angle penalty: 0 if aligned (cos=1), 2 if opposite (cos=-1)
        angle_penalty = 1.0 - cos_theta
        
        # Distance penalty: Normalize by the sum of radii to keep scale consistent
        # This keeps dist_penalty in a comparable range to angle_penalty
        norm_factor = a + b + 1e-12
        dist_penalty = c / norm_factor
        
        # Weighted combination:
        # Strong preference for distance (proximity)
        # Moderate penalty for angular deviation (backtracking)
        score = 1.0 * dist_penalty + 0.8 * angle_penalty
        
        if score < best_score:
            best_score = score
            best_node = node
    
    return int(best_node)
# EVOLVE-BLOCK-END
