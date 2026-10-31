"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step Lookahead Nearest Neighbor heuristic.
    
    For each candidate node j, we estimate the cost of moving to j and then to
    its nearest neighbor k among the remaining unvisited nodes. If j is the last
    unvisited node, the cost includes the return to the destination.
    This approach balances immediate proximity with future connectivity,
    avoiding dead-ends that pure Nearest Neighbor might fall into.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        # Cost 1: current -> n1 -> n2 -> dest
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost 2: current -> n2 -> n1 -> dest
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2

    # General case: n_unvisited > 2
    # We want to pick j in unvisited_nodes that minimizes:
    # d(current, j) + d(j, next(j))
    # where next(j) is the nearest neighbor of j in unvisited_nodes \ {j}
    # If j is the only one left (not the case here), we'd add d(j, dest).
    # But since n_unvisited > 2, there is always a next(j).
    
    best_node = -1
    best_cost = np.inf
    
    # Precompute distances from current to all unvisited
    # and distances between unvisited nodes
    # To be efficient, we can iterate through unvisited nodes
    
    for j_idx in range(n_unvisited):
        j = int(unvisited_nodes[j_idx])
        
        # Cost to reach j from current
        cost_to_j = distance_matrix[current_node, j]
        
        # If cost_to_j is already greater than best_cost, we can't improve 
        # because d(j, next(j)) >= 0
        if cost_to_j >= best_cost:
            continue
            
        # Find nearest neighbor of j among unvisited nodes excluding j
        # We need min distance from j to any k in unvisited_nodes where k != j
        # We can use the distance_matrix row for j
        
        # Get distances from j to all unvisited nodes
        # unvisited_nodes is an array of node indices
        dists_from_j = distance_matrix[j, unvisited_nodes]
        
        # Set distance to j itself to inf to ignore it
        # j is at index j_idx in unvisited_nodes
        dists_from_j[j_idx] = np.inf
        
        # The nearest neighbor is the argmin
        next_j_idx = np.argmin(dists_from_j)
        d_j_next = dists_from_j[next_j_idx]
        
        total_cost = cost_to_j + d_j_next
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = j
            
    return best_node
# EVOLVE-BLOCK-END
