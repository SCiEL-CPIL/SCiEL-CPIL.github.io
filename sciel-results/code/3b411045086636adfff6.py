"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    For each candidate next node, it estimates the cost of moving to it 
    and then to the nearest subsequent unvisited node (or destination if last).
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen if called correctly, but safety check
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Get distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # We need to evaluate a score for each candidate in unvisited_nodes
    # Score for candidate i: dist(current, i) + dist(i, best_next(i))
    # where best_next(i) is the closest unvisited node to i, 
    # unless i is the last one, then it's distance to destination.
    
    # Optimization: Precompute distances between unvisited nodes?
    # unvisited_nodes is an array of indices.
    # dist_matrix_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # This is O(N^2) memory and time per step. For N=50, it's 2500 elements. Very fast.
    
    # Let's construct the sub-matrix of distances between unvisited nodes
    # This allows us to find nearest neighbors efficiently.
    unvisited_indices = unvisited_nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)]
    
    # Find the nearest neighbor for each node in the sub_matrix
    # Exclude self by setting diagonal to inf
    np.fill_diagonal(sub_matrix, np.inf)
    nearest_neighbor_indices = np.argmin(sub_matrix, axis=1)
    nearest_neighbor_dists = np.min(sub_matrix, axis=1)
    
    # Now, for each candidate i in unvisited_nodes:
    # If i is the last node (index n_unvisited - 1 in the logical order? No, order doesn't matter for "last" in terms of set, but the problem implies a specific destination).
    # Wait, the "destination" is fixed. The "last" unvisited node is the one that will connect to destination_node.
    # However, we don't know which one will be last until we pick all others.
    # The heuristic is: if we pick i, what is the likely next cost?
    # If i is the *only* remaining node after this pick, the next cost is dist(i, destination).
    # But we are picking *one* node now. There are n_unvisited - 1 left.
    # So the "next" node after i is chosen from the remaining n_unvisited - 1 nodes.
    # The standard lookahead uses the closest *other* unvisited node.
    # If only 2 nodes remain (current + 2 unvisited), and we pick one, the other is the last.
    # If n_unvisited == 2:
    #   Candidate A: cost = dist(C, A) + dist(A, B) (where B is the other one, which will be last, so actually dist(A, D) if B is last? No, if we pick A, B is last. So next step is B, then D. The lookahead for A is dist(C, A) + dist(A, B).
    #   Candidate B: cost = dist(C, B) + dist(B, A).
    #   This works.
    
    # General case:
    # For candidate i, the estimated next cost is dist(i, nearest_neighbor(i)).
    # However, if i is the node that *will be* the last one, the next cost should be dist(i, destination).
    # But we don't know which one is last yet.
    # A common refinement: 
    # If n_unvisited == 2, we must compare:
    #   Option 1: Go to A, then B, then D. Cost = d(C,A) + d(A,B) + d(B,D)
    #   Option 2: Go to B, then A, then D. Cost = d(C,B) + d(B,A) + d(A,D)
    #   We should pick the one with lower total.
    #   This is equivalent to:
    #   Score A = d(C,A) + d(A,B)
    #   Score B = d(C,B) + d(B,A)
    #   But wait, the final leg d(B,D) or d(A,D) is not included in the "next step" cost in the simple lookahead.
    #   To be more accurate, we should include the destination cost if the candidate is likely to be last.
    #   Actually, a better lookahead for the last 2 nodes is to just calculate the total remaining tour cost for both permutations and pick the best.
    
    if n_unvisited == 2:
        node1 = unvisited_indices[0]
        node2 = unvisited_indices[1]
        
        # Option 1: node1 then node2
        cost1 = distance_matrix[current_node, node1] + distance_matrix[node1, node2] + distance_matrix[node2, destination_node]
        # Option 2: node2 then node1
        cost2 = distance_matrix[current_node, node2] + distance_matrix[node2, node1] + distance_matrix[node1, destination_node]
        
        if cost1 < cost2:
            return int(node1)
        else:
            return int(node2)

    # For n_unvisited > 2:
    # Score for candidate i: dist(C, i) + dist(i, nearest_unvisited(i))
    # We have nearest_neighbor_dists calculated.
    # We also have dists_from_current.
    
    scores = dists_from_current + nearest_neighbor_dists
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_indices[best_idx])

# EVOLVE-BLOCK-END
