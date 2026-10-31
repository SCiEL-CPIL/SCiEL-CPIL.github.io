import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        c1 = unvisited_nodes[0]
        c2 = unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1) if cost1 <= cost2 else int(c2)

    # Extract distances
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Compute average distance from each candidate to all other unvisited nodes (connectivity/isolation proxy)
    # This is O(N^2) per call, which is acceptable for N=50.
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    avg_dist = sub_dist.sum(axis=1) / (n_unvisited - 1)
    
    # Determine destination bias weight
    # Ramp up destination bias as we get closer to the end of the tour.
    # progress = 0.0 at start (50 unvisited), 1.0 at end (0 unvisited, but we stop at 1)
    progress = 1.0 - (n_unvisited / 50.0)
    
    # Use a stronger ramp for destination bias
    # Start with a mild bias, increase significantly later
    dest_weight = 0.2 + 0.8 * (progress ** 2)
    
    # Connectivity weight: encourage staying close to the cluster of unvisited nodes
    # This helps avoid long jumps and keeps the tour compact
    conn_weight = 0.15
    
    # Combine scores
    # Lower score is better
    scores = dist_from_current + dest_weight * dist_to_dest + conn_weight * avg_dist
    
    # Find min
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
