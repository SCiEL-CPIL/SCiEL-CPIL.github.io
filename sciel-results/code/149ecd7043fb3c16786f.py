"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return time.time() >= _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a minimax-inspired centrality heuristic.
    
    The key insight: instead of minimizing cumulative cost, we minimize the
    maximum distance to other unvisited cities. This prevents any single
    long edge from dominating the tour length by favoring central nodes
    that are not too far from any remaining city.
    
    Combined with a mild nearest-neighbor bias to maintain local coherence.
    """
    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if close to limit, fall back to fast nearest-neighbor
    if _deadline_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n = len(unvisited_nodes)
    
    # Get distances from current node to all unvisited nodes
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from unvisited nodes to destination
    d_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute the distance matrix among unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Minimax centrality: for each candidate i, compute the maximum distance
    # from i to any other unvisited node j (j != i).
    # This measures how "central" node i is among the remaining nodes.
    # A node with low max-distance is well-connected to all others.
    # We mask the self-distance by setting diagonal to 0 before taking max.
    d_no_self = d_unvisited.copy()
    np.fill_diagonal(d_no_self, 0.0)
    
    # Max distance from each candidate to any other unvisited node
    max_dist_to_others = np.max(d_no_self, axis=1)
    
    # Also compute a softer measure: second-max or high percentile to be robust
    # Use the 90th percentile of distances to others as a robust "bottleneck"
    if n > 2:
        # For each row, get sorted distances and take a high percentile
        sorted_dists = np.sort(d_no_self, axis=1)
        # Take the value at index int(0.9 * (n-1)) as a robust max
        idx_90 = int(0.9 * (n - 1))
        bottleneck_90 = sorted_dists[:, idx_90]
    else:
        bottleneck_90 = max_dist_to_others
    
    # Normalize all components to [0, 1]
    d_from_current_norm = d_from_current / (np.max(d_from_current) + 1e-10)
    d_to_dest_norm = d_to_dest / (np.max(d_to_dest) + 1e-10)
    max_dist_norm = max_dist_to_others / (np.max(max_dist_to_others) + 1e-10)
    bottleneck_90_norm = bottleneck_90 / (np.max(bottleneck_90) + 1e-10)
    
    # Combine: 
    # - Primary: minimize bottleneck (max distance to others) — prevents long edges
    # - Secondary: mild nearest-neighbor bias for local coherence
    # - Tertiary: destination proximity to help close the tour
    w_bottleneck = 0.50
    w_nn = 0.30
    w_dest = 0.20
    
    score = (w_bottleneck * max_dist_norm + 
             w_nn * d_from_current_norm + 
             w_dest * d_to_dest_norm)
    
    # Select the node with minimum combined score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
