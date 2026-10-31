"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to maintain the "tree" context if we were doing a full Prim's, 
# but since the interface is stateless (we only get current_node and unvisited), 
# we must approximate the MST growth using only the provided inputs.
#
# The directive suggests: "minimizing the distance to the current MST (formed by visited nodes) 
# plus a penalty proportional to the increase in the MST's total weight".
#
# In a standard constructive TSP (like Nearest Neighbor), we build a path. 
# Here, we frame it as growing a tree. 
# However, we don't have the full history of visited nodes, only `current_node` and `unvisited_nodes`.
# This implies the "MST" or "structure" is implicitly defined by the path built so far.
#
# A robust structural change that fits "MST extension" in a stateless interface:
# Instead of just looking at the immediate edge (current -> candidate), we look at 
# how well the candidate integrates into the *remaining* cluster to form a low-cost 
# spanning structure, while still connecting from the current position.
#
# Let's use a "Steiner-like" or "Prim-like" step:
# 1. Calculate the cost to reach the candidate from the current node: C_reach = dist(current, c)
# 2. Calculate the cost to integrate the candidate into the rest of the unvisited nodes.
#    In Prim's algorithm, we add the node with the minimum edge to the *current tree*.
#    Here, our "current tree" is effectively the set of visited nodes, but we only know the root (current).
#    However, we can approximate the "integration cost" by looking at the minimum distance 
#    from the candidate to any OTHER unvisited node. This represents the "cheapest way to continue" 
#    from that candidate into the remaining cluster.
#    
#    Score(c) = dist(current, c) + alpha * min_{u in unvisited, u!=c} dist(c, u)
#
# This is different from the current bottleneck strategy (which used MAX distance).
# Using MIN distance encourages choosing nodes that are well-connected to the rest of the 
# remaining cluster, similar to how Prim's algorithm expands the MST by the cheapest edge.
# This avoids "dead ends" where a node is far from everyone else.

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using an MST-growth (Prim-like) heuristic.
    
    Strategy:
    We treat the construction as expanding a spanning structure. 
    For each candidate node c in unvisited_nodes:
    1. Immediate Cost: dist(current_node, c). This is the edge we are adding to the path.
    2. Integration Cost: The minimum distance from c to any other unvisited node. 
       This approximates the cost of the next step in a Prim's algorithm expansion 
       from the "frontier". It penalizes candidates that are isolated from the rest 
       of the unvisited cluster.
       
    We minimize the sum of these two costs. This shifts the focus from pure 
    local proximity (NN) or local extremity (Bottleneck) to structural connectivity.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_remaining = len(unvisited_nodes)
    
    # 1. Distance from current to each candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each candidate, find the minimum distance to any OTHER unvisited node.
    # We need the sub-matrix of distances between unvisited nodes.
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # The diagonal is 0. We want the min distance to *other* nodes.
    # We can set the diagonal to infinity before taking the min to exclude self.
    # np.fill_diagonal(sub_matrix, np.inf) modifies the view, so we copy or use a trick.
    # Trick: add a large number to diagonal? No, just copy and fill.
    # For n=50, copying is cheap.
    sub_matrix_copy = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_copy, np.inf)
    
    # min_dist_to_other is the minimum edge weight connecting c to the rest of the unvisited set.
    min_dist_to_other = np.min(sub_matrix_copy, axis=1)
    
    # Handle edge case where n_remaining == 1 (already handled above)
    # If n_remaining > 1, min_dist_to_other will be a valid distance.
    
    # 3. Combine scores
    # We balance the immediate cost with the integration cost.
    # A weight of 1.0 suggests equal importance, but we might want to prioritize 
    # the immediate step slightly less if the integration is bad? 
    # Or prioritize integration to avoid dead ends?
    # Let's try a weight of 0.5 for the integration term to start, 
    # similar to the previous bottleneck weight, but the metric is different (min vs max).
    
    w_integration = 0.5
    scores = dists_from_current + w_integration * min_dist_to_other
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
