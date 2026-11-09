"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a connectivity-aware heuristic that balances:
    1. Distance from current node to candidate
    2. Distance from candidate to destination (lookahead)
    3. Penalty for candidates that would create long gaps in the tour
    
    This prevents disconnected subtours and ensures efficient tour closure.
    """
    if _check_deadline():
        # Fallback to nearest neighbor if deadline approaching
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Compute base distances from current node to all candidates
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from candidates to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute a "gap penalty": for each candidate, estimate how much it increases
    # the maximum edge in the partial tour by looking at neighbors in the candidate set
    # This is an approximation of maintaining connectivity
    
    # For efficiency, use a simpler heuristic: penalize candidates that are far
    # from both current and destination (creates isolated points)
    # Weighted score: local distance + alpha * lookahead + beta * isolation penalty
    
    alpha = 0.3  # Weight for lookahead to destination
    beta = 0.1   # Weight for isolation penalty
    
    # Isolation penalty: how far is the candidate from its nearest unvisited neighbor?
    # This approximates whether choosing this node creates a gap in the tour
    # Compute pairwise distances among unvisited nodes (for small n this is fine)
    if n <= 100:
        # Compute min distance from each candidate to any other unvisited node
        # This helps avoid picking isolated points
        candidate_distances = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Set diagonal to inf so we don't consider self-distance
        np.fill_diagonal(candidate_distances, np.inf)
        min_neighbor_dist = np.min(candidate_distances, axis=1)
        # Normalize: penalize candidates that are far from their nearest neighbor
        # Scale by the median distance to make it relative
        median_dist = np.median(min_neighbor_dist) if np.any(min_neighbor_dist > 0) else 1.0
        isolation_penalty = (min_neighbor_dist - median_dist) / (median_dist + 1e-10)
        isolation_penalty = np.maximum(isolation_penalty, 0)  # Only penalize if above median
    else:
        # For larger sets, use a simpler approximation
        isolation_penalty = np.zeros(n)
    
    # Combine scores
    # Normalize each component to [0, 1] range for better weighting
    max_dist_current = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
    max_dist_dest = np.max(dist_to_dest) if np.max(dist_to_dest) > 0 else 1.0
    max_isolation = np.max(isolation_penalty) if np.max(isolation_penalty) > 0 else 1.0
    
    score = (dist_from_current / max_dist_current + 
             alpha * (dist_to_dest / max_dist_dest) + 
             beta * (isolation_penalty / max_isolation))
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
