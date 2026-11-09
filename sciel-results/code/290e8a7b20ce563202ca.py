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

    # 2. Compute the centroid of the feasible nodes
    # We use the geometric mean of the node indices? No, we need coordinates.
    # Wait, the input is `distance_matrix`, not coordinates.
    # We cannot compute a geometric centroid in Euclidean space directly from a distance matrix 
    # unless we embed it or use a proxy.
    # 
    # Re-reading the directive: "density-weighted centroid of all feasible unvisited nodes".
    # In the context of a distance matrix only, a "centroid" is tricky.
    # However, we can use the **median** or **mean** of the distances? No.
    # 
    # Alternative interpretation: 
    # The "centroid" in a graph metric space can be approximated by the node 
    # that minimizes the sum of distances to all other feasible nodes (the 1-median).
    # Or, simpler: The node that is "closest to the center" of the cluster.
    #
    # Given N=50, computing the 1-median (minimizing sum of distances) for each step is O(N^2) 
    # per selection, which is 50*50 = 2500 ops per step. 50 steps -> 125,000 ops. 
    # This is very fast in Python/NumPy.
    #
    # Let's compute the "center" of the feasible cluster as the node that minimizes 
    # the total distance to all other feasible nodes.
    
    # Get distance matrix for feasible nodes
    # feasible_nodes is an array of indices
    # dist_sub = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Sum of distances from each feasible node to all other feasible nodes
    # This identifies the "hub" or "center" of the feasible set.
    sub_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    sum_dists = np.sum(sub_dist, axis=1)
    
    # The index in feasible_nodes with the minimum sum of distances is the "centroid" node
    # Note: This is a discrete centroid (1-median).
    centroid_idx_in_feasible = np.argmin(sum_dists)
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
    # Weighting: Immediate distance is crucial. Centroid distance helps with global structure.
    # Let's try a linear combination.
    # Score = dist_to_current + 0.5 * dist_to_centroid
    
    # To make it scale-invariant or robust, we might normalize, but simple sum works well for TSP/CVRP heuristics.
    # Let's use a weight of 0.5 for the centroid term.
    scores = dist_to_current + 0.5 * dist_to_centroid
    
    # Pick the node with the minimum score
    best_idx_in_feasible = np.argmin(scores)
    next_node = feasible_nodes[best_idx_in_feasible]
    
    return int(next_node)

# EVOLVE-BLOCK-END
