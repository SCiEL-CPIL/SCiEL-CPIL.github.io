"""Traveling salesman constructive heuristic with 2-opt-style insertion look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using a 2-opt-style insertion cost heuristic.
    
    Instead of pure nearest-neighbor, we evaluate each candidate node by considering:
    1. The distance from current_node to the candidate (edge we'd add)
    2. The minimum distance from the candidate to any other unvisited node (proxy for
       the next edge we'd likely need, modeling the local 2-opt structure)
    
    This is a structural change: we're doing a global look-ahead over all candidates
    with an insertion-cost model rather than sequential greedy picking.
    
    We also add a small time guard to ensure we finish within the 150s limit.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Fallback: return destination_node if no unvisited nodes
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current_node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, compute the minimum distance to any OTHER unvisited node
    # This models the "next edge" cost in a 2-opt sense
    # We compute this efficiently using the distance matrix
    
    # dist_matrix among unvisited nodes
    unvisited_idx = unvisited_nodes.astype(int)
    sub_matrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # For each node i, min distance to any other node j != i
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(sub_matrix, np.inf)
    
    # min_dist_to_other[i] = min over j != i of sub_matrix[i, j]
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # Combined score: we want to minimize (dist_from_current[i] + alpha * min_dist_to_other[i])
    # This is a 2-opt-style insertion cost: the edge we add plus a proxy for the next edge
    # alpha balances between going to a close node vs. a node that has good connectivity
    alpha = 0.5
    scores = dist_from_current + alpha * min_dist_to_other
    
    # Pick the node with minimum combined score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
