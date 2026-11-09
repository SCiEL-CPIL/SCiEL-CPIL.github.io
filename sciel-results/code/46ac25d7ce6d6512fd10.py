"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit based on a density-weighted centroid heuristic.
    
    Args:
        current_node (int): The index of the current node.
        depot (int): The index of the depot (usually 0).
        unvisited_nodes (list or np.array): List of unvisited node indices.
        rest_capacity (float): Remaining capacity of the vehicle.
        demands (list or np.array): Demand for each node.
        distance_matrix (np.array): NxN matrix of distances.
        
    Returns:
        int: The index of the next node to visit, or depot (0) to close the route.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy arrays for fast operations if they aren't already
    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter for feasible nodes (capacity constraint)
    # demands is likely an array aligned with node indices
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. Compute the demand-weighted centroid of the feasible nodes
    # The "density" is interpreted as the demand of the nodes.
    # We compute the node that minimizes the demand-weighted sum of distances to all other feasible nodes.
    # This is the demand-weighted 1-median.
    
    # Get distance matrix for feasible nodes
    sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Get demands for feasible nodes
    feasible_demands = demands[feasible_nodes]
    
    # Compute demand-weighted sum of distances for each feasible node
    # weighted_dists[i] = sum_j (demand[j] * dist[i][j])
    weighted_dists = sub_dist @ feasible_demands
    
    # The index in feasible_nodes with the minimum weighted sum is the "centroid" node
    centroid_idx_in_feasible = np.argmin(weighted_dists)
    centroid_node = feasible_nodes[centroid_idx_in_feasible]
    
    # 3. Score candidates
    # We want to pick a node that is:
    # a) Close to current_node (immediate cost)
    # b) Close to the centroid_node (maintains cluster density / reduces future spread)
    
    # Distances from current_node to all feasible nodes
    dist_to_current = distance_matrix[current_node, feasible_nodes]
    
    # Distances from centroid_node to all feasible nodes
    dist_to_centroid = distance_matrix[centroid_node, feasible_nodes]
    
    # Combine scores.
    # Use a slightly higher weight for the centroid term to enforce tighter clustering
    # as suggested by "minimize intra-cluster travel variance".
    scores = dist_to_current + 0.8 * dist_to_centroid
    
    # Pick the node with the minimum score
    best_idx_in_feasible = np.argmin(scores)
    next_node = feasible_nodes[best_idx_in_feasible]
    
    return int(next_node)

# EVOLVE-BLOCK-END
