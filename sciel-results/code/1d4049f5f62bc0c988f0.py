"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Regret-2 heuristic: For each feasible customer, compute the difference
    between the distance to its second-nearest feasible neighbor and its nearest
    feasible neighbor. Prioritize nodes with high regret (unique close connections)
    to avoid breaking up efficient pairs/clusters. Tie-break or bias towards
    nodes closer to the current position to minimize total route length.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Get distances from current_node to all unvisited nodes (for later bias)
    # First, filter to feasible nodes
    
    # Ensure unvisited_nodes is a numpy array for easy indexing
    unvisited_arr = np.asarray(unvisited_nodes)
    
    # Filter to only feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        # No feasible customers left, go back to depot
        return depot

    # Get the feasible unvisited node indices (original node IDs)
    feasible_nodes = unvisited_arr[feasible_indices]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Get the sub-matrix of distances between feasible nodes
    # distance_matrix is assumed to be a numpy array
    sub_dm = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    n = len(feasible_nodes)
    regrets = np.zeros(n)
    
    # Calculate regret for each node
    for i in range(n):
        if n > 2:
            dists = sub_dm[i, :].copy()
            dists[i] = np.inf  # Exclude self
            # Find the two smallest distances
            # Using argpartition for efficiency, though n is small
            # np.partition returns array with kth smallest element in kth position
            # We want indices of 2 smallest
            if n > 2:
                # Get indices of 2 smallest
                # Note: np.argpartition returns indices, not sorted values
                part_indices = np.argpartition(dists, 2)[:2]
                # Sort these two to get min1 and min2
                min1 = dists[part_indices[0]]
                min2 = dists[part_indices[1]]
                # Ensure min1 <= min2
                if min1 > min2:
                    min1, min2 = min2, min1
                regrets[i] = min2 - min1
            else:
                regrets[i] = 0
        else:
            regrets[i] = 0

    # Calculate distance from current node to each feasible node
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Combine regret and proximity
    # High regret is good (maximize), high distance is bad (minimize)
    # Score = Regret - lambda * Distance
    # We use lambda = 0.5 to strongly prefer nearby nodes if regret is not overwhelming
    lambda_weight = 0.5
    scores = regrets - lambda_weight * dists_from_current
    
    # Pick the node with the highest score
    best_idx = np.argmax(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
