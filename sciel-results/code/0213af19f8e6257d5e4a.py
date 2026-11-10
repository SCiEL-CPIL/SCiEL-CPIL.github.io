"""Improved angular sweep heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using an angular sweep heuristic around the depot.
    
    This heuristic minimizes the angular deviation between the vector (Depot -> Current)
    and (Depot -> Candidate), while penalizing the distance from the current node.
    This encourages the route to sweep around the depot in a consistent sector,
    reducing backtracking and route length.
    """
    depot = int(depot)
    
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
        
        # Compute angle at depot using law of cosines
        # a = d(depot, current), b = d(depot, node), c = d(current, node)
        a = d_depot_current
        b = d_depot_node
        c = d_current_node
        
        if a < 1e-12 or b < 1e-12:
            angle = 0.0
        else:
            cos_theta = (a*a + b*b - c*c) / (2.0 * a * b)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle = np.arccos(cos_theta)
        
        # New Scoring:
        # 1. Angular penalty: Minimize the angle to stay in the same sector.
        # 2. Distance penalty: Minimize the actual travel distance from current node.
        # We normalize the distance by the sum of the radii (a+b) to keep it scale-invariant
        # relative to the position in the sector, but weight it more heavily than before
        # to avoid picking distant nodes that are only slightly off-angle.
        
        # Using a weighted sum:
        # score = w_angle * angle + w_dist * (c / (a + b + eps))
        # Let's try w_angle = 1.0, w_dist = 0.25
        
        denom = a + b + 1e-12
        dist_penalty = c / denom
        score = 1.0 * angle + 0.25 * dist_penalty
        
        if score < best_score:
            best_score = score
            best_node = node
    
    return int(best_node)
# EVOLVE-BLOCK-END
