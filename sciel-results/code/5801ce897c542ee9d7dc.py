"""Potential-field TSP constructive heuristic with adaptive repulsion."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 112.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a potential field model.
    
    The score combines:
    1. Attractive force: distance from current node (immediate cost)
    2. Repulsive force: inverse-distance potential from other unvisited nodes
       (encourages spatial coverage, prevents clustering)
    3. Destination awareness: slight penalty for nodes far from destination
       (helps close the tour)
    
    For small sets (<=5), use exact brute force for optimal tail.
    """
    n_unvisited = len(unvisited_nodes)
    
    # If only 1 node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If few nodes left, use exact brute force to optimize the tail
    if n_unvisited <= 5:
        nodes = list(unvisited_nodes)
        best_cost = np.inf
        best_first_node = nodes[0]
        
        for perm in permutations(nodes):
            # Cost: current -> perm[0] -> ... -> perm[-1] -> destination
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = perm[0]
                
        return int(best_first_node)
    
    # Check deadline - if close to limit, use fast nearest neighbor
    if _is_deadline_passed():
        dist_curr = distance_matrix[current_node, unvisited_nodes]
        idx = np.argmin(dist_curr)
        return int(unvisited_nodes[idx])
    
    # Potential field approach
    # 1. Distance from current to all unvisited (attractive)
    dist_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each unvisited to destination (destination awareness)
    dist_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Repulsive potential: sum of inverse distances to other unvisited nodes
    #    This creates a "pressure" that pushes selection toward spatially isolated nodes
    D_ii = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Exclude self by setting diagonal to inf
    np.fill_diagonal(D_ii, np.inf)
    
    # Inverse distance potential (repulsion)
    # Use a smoothed inverse to avoid numerical issues
    inv_dist = 1.0 / (D_ii + 1e-10)
    
    # Sum of repulsive potential from all other unvisited nodes
    repulsion = np.sum(inv_dist, axis=1)
    
    # Normalize repulsion to be comparable to distances
    # Scale by the median distance to make it dimensionally consistent
    med_dist = np.median(dist_curr)
    repulsion_scaled = repulsion * med_dist * 0.1  # Tuning factor
    
    # Destination penalty: nodes far from destination get a small penalty
    # This helps ensure the tour can close efficiently
    dest_penalty = dist_dest * 0.1
    
    # Combined score
    # Primary: distance from current (immediate edge cost)
    # Secondary: repulsion (spatial coverage)
    # Tertiary: destination awareness (tour closure)
    scores = dist_curr + repulsion_scaled + dest_penalty
    
    # Select the node with the minimum score
    idx = np.argmin(scores)
    
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
