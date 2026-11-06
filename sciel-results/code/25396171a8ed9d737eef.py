"""Improved TSP constructive heuristic using local density optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = time.time()
_DEADLINE_MARGIN = 2.0  # seconds before the 150s limit

def _check_deadline():
    """Return True if we should stop searching."""
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city for a traveling-salesman tour using a local density
    optimization heuristic that considers nearest-neighbor gap metrics.
    
    Parameters:
        current_node: int, the node we are currently at
        destination_node: int, the final destination (not directly used in heuristic)
        unvisited_nodes: np.ndarray of unvisited node indices
        distance_matrix: np.ndarray, pairwise distance matrix
    
    Returns:
        int, the index of the next node to visit
    """
    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if tight, fall back to simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each unvisited node, compute a local density score:
    # - How close is this node to its nearest neighbor among unvisited nodes?
    # - How close is this node to the destination? (to avoid long detours)
    # - Balance between immediate proximity and future connectivity
    
    # Compute for each candidate:
    # 1. Distance from current (immediate cost)
    # 2. Distance to nearest other unvisited node (local connectivity)
    # 3. Distance to destination (final return consideration)
    
    # Vectorized computation of nearest-neighbor gap for each unvisited node
    # For each unvisited node i, find min distance to other unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Nearest neighbor distance for each unvisited node
    nn_dist = np.min(sub_matrix, axis=1)
    
    # Distance to destination for each unvisited node
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combine scores:
    # Primary: distance from current node
    # Secondary: penalty for isolated nodes (high nn_dist means poor local connectivity)
    # Tertiary: mild penalty for being far from destination (to avoid bad endings)
    
    # Normalize components to [0, 1] range for weighting
    max_dist_current = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
    max_nn = np.max(nn_dist) if np.max(nn_dist) > 0 else 1.0
    max_dest = np.max(dist_to_dest) if np.max(dist_to_dest) > 0 else 1.0
    
    norm_current = dist_from_current / max_dist_current
    norm_nn = nn_dist / max_nn
    norm_dest = dist_to_dest / max_dest
    
    # Weighted combination:
    # - 0.6 weight on immediate distance (most important)
    # - 0.25 weight on local connectivity (prefer well-connected regions)
    # - 0.15 weight on distance to destination (mild preference)
    score = 0.6 * norm_current + 0.25 * norm_nn + 0.15 * norm_dest
    
    # Pick the node with minimum score
    best_idx = int(np.argmin(score))
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
