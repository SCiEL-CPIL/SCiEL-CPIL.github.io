"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour.
    
    Uses a combined heuristic that balances:
    1. Distance to the current node (local optimality)
    2. Distance to the destination node (global structure)
    3. Avoiding isolated/unvisited cities that are far from everything
    
    The scoring is on mean closed-tour length, so we want to minimize total tour length.
    A good constructive heuristic should avoid creating long edges and maintain a balanced tour.
    
    Strategy: For each candidate node, compute a combined score that considers:
    - The edge from current to candidate
    - The edge from candidate to destination (or to the nearest unvisited if destination is already visited)
    - A penalty for candidates that would leave some unvisited city very far from the current path
    
    We use a weighted sum where we prioritize keeping the tour compact.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # If we have only two unvisited nodes, pick the one that minimizes the total
    # edge cost: current -> node -> destination
    if len(unvisited_nodes) == 2:
        n1, n2 = unvisited_nodes
        # Cost if we go to n1 first: dist(current, n1) + dist(n1, n2) + dist(n2, dest)
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 first: dist(current, n2) + dist(n2, n1) + dist(n1, dest)
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
    
    # For general case, use a scoring function that considers:
    # 1. Distance from current to candidate
    # 2. Distance from candidate to destination (to maintain path toward destination)
    # 3. A look-ahead term: how well does this candidate connect to the remaining unvisited nodes?
    
    # Vectorized computation for speed
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Look-ahead: for each candidate, compute the minimum distance from that candidate
    # to any other unvisited node (excluding itself). This helps avoid picking a node
    # that would be isolated.
    # We want to minimize the distance from candidate to the rest of the unvisited set.
    # A simple proxy: the average distance from candidate to all other unvisited nodes,
    # or the minimum distance (to avoid isolation).
    
    # Compute pairwise distances among unvisited nodes
    # For efficiency, we compute the minimum distance from each unvisited node to any other unvisited node
    # This can be done by taking the min of each row of the submatrix, excluding the diagonal
    
    sub_dist = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(sub_dist, np.inf)
    # Use max distance to other unvisited nodes as a "minimax" style connectivity penalty
    # This encourages picking nodes that are central to the remaining set
    max_dist_to_other_unvisited = np.max(sub_dist, axis=1)
    
    # Combine scores:
    # Primary: distance from current to candidate
    # Secondary: distance from candidate to destination (weighted less)
    # Tertiary: prefer candidates that are central to the remaining unvisited set (low max_dist)
    
    # Normalize terms to comparable scales
    # Use a weighted sum with tuned weights
    weight_current = 1.0
    weight_dest = 0.3
    weight_connectivity = 0.5
    
    score = (weight_current * dist_from_current + 
             weight_dest * dist_to_dest + 
             weight_connectivity * max_dist_to_other_unvisited)
    
    # Pick the candidate with the minimum combined score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
