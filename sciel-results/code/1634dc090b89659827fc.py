"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Greedy with Return heuristic.
    
    This approach evaluates each candidate node based on the sum of the distance
    from the current node to the candidate and the distance from the candidate 
    to the destination node. This balances the immediate cost of moving to the 
    candidate with the anticipated cost of returning to the destination, 
    preventing the algorithm from getting stuck in areas that are far from 
    the destination.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        # Exact evaluation for the last two nodes
        # Path: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Path: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2

    # Distance from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from all unvisited nodes to the destination node
    dist_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Heuristic Score:
    # Sum of cost to reach candidate and cost to return from candidate.
    # This encourages picking nodes that are not only close to the current
    # position but also close to the destination, ensuring a balanced tour.
    
    scores = dist_from_current + dist_to_destination
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
