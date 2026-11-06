import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return destination_node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Find the "median" of the unvisited nodes
    # The median is the node that minimizes the sum of distances to all other unvisited nodes.
    # This approximates the geometric center of the remaining cluster.
    # We use this to identify "peripheral" nodes which are good for TSP tours.
    
    # Vectorized computation of sum of distances from each unvisited node to all others
    # dist_matrix_subset is of shape (N_unvisited, N_unvisited)
    dist_subset = distance_matrix[unvisited_nodes, unvisited_nodes]
    sum_dists = np.sum(dist_subset, axis=1)
    
    # The index of the median node in the unvisited_nodes array
    median_local_idx = np.argmin(sum_dists)
    median_node = unvisited_nodes[median_local_idx]
    
    # Calculate distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate distances from each unvisited node to the median node
    # This represents how "central" or "peripheral" the node is.
    # Larger distance to median means more peripheral.
    dist_to_median = distance_matrix[unvisited_nodes, median_node]
    
    # Heuristic: Minimize distance from current, but reward being far from the median (peripheral).
    # The coefficient 0.5 balances the immediate cost against the structural benefit of staying on the hull.
    # Tuning this coefficient is key. 
    # A value of 0 reduces to Nearest Neighbor.
    # A value of 1 strongly prefers the most peripheral reachable node.
    
    alpha = 0.5
    scores = dist_from_current - alpha * dist_to_median
    
    best_local_idx = np.argmin(scores)
    best_node = unvisited_nodes[best_local_idx]
    
    return int(best_node)
