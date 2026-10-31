"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a multi-criteria heuristic that balances:
    1. Distance from current node (nearest neighbor component)
    2. Distance from destination node (to avoid long detours)
    3. A penalty for nodes that would create a 'dead end' (high cost to connect back)
    
    This is a greedy look-ahead heuristic that considers the cost of adding
    a node to the tour and the eventual cost of closing the tour.
    """
    if len(unvisited_nodes) == 0:
        return -1  # Should not happen if called correctly
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if len(unvisited_nodes) == 2:
        # For two remaining nodes, pick the one that minimizes the total cost
        # of going current -> node -> other_node -> destination
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination node to all unvisited nodes
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # For each candidate node, estimate the cost if we pick it:
    # - Immediate cost: distance from current to candidate
    # - Future cost heuristic: after picking candidate, we'll still need to visit
    #   the remaining nodes and return to destination.
    #   A simple proxy: the minimum distance from candidate to any other unvisited node
    #   plus the minimum distance from any unvisited node to destination.
    
    # To keep it efficient and simple, use a weighted combination:
    # score = alpha * dist_from_current + beta * dist_to_dest
    
    # The idea: we want to stay close to current (greedy) but also not pick a node
    # that is very far from the destination, as that might create a long return path.
    
    # Adaptive weights: when few nodes remain, care more about destination distance.
    # When many nodes remain, care more about current distance.
    n_unvisited = len(unvisited_nodes)
    
    # Weight for destination distance increases as we get closer to the end
    # This helps avoid picking a node that is close to current but far from destination
    # when only a few nodes are left.
    dest_weight = 0.3 + 0.7 * (1.0 - n_unvisited / 50.0)
    current_weight = 1.0 - dest_weight
    
    # Normalize distances to [0, 1] range for better combination
    dist_from_current_norm = dist_from_current / (np.max(dist_from_current) + 1e-10)
    dist_to_dest_norm = dist_to_dest / (np.max(dist_to_dest) + 1e-10)
    
    # Combined score: lower is better
    scores = current_weight * dist_from_current_norm + dest_weight * dist_to_dest_norm
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
