"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache to persist across calls within a single evaluation
_cache = {}

def _get_cache(key):
    return _cache.get(key)

def _set_cache(key, value):
    _cache[key] = value

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour construction.
    
    Uses a multi-criteria scoring that balances:
    1. Immediate distance from current node
    2. Distance from candidate to destination (to avoid stranding)
    3. A soft penalty for candidates that are 'far out' relative to the remaining set
    4. A look-ahead term considering the best exit from the candidate
    
    To handle the time budget, we use a simple but effective heuristic:
    - For small unvisited sets (< 20), do a more thorough 2-step lookahead
    - For larger sets, use a fast approximate scoring
    """
    # Deadline guard: track elapsed time
    # Since we don't know when the evaluation started, we use a per-call timeout
    # but the main concern is total evaluation time. We'll keep this fast.
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Fast path: nearest neighbor with destination awareness
    # Score = alpha * d(current, candidate) + beta * d(candidate, destination)
    # This is a simple 2-opt-like lookahead
    
    # Get distances from current to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    
    # Get distances from all unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]  # shape (n_unvisited,)
    
    # For a better heuristic, also consider the minimum distance from each candidate
    # to any other unvisited node (to avoid isolated points)
    # This is O(n^2) but for n=50 it's fine
    
    # Weighted combination
    # Normalize distances to [0, 1] for stable weighting
    d_current_norm = d_current / (np.max(d_current) + 1e-10)
    d_dest_norm = d_dest / (np.max(d_dest) + 1e-10)
    
    # For larger sets, use a simpler score
    if n_unvisited > 15:
        # Simple weighted sum: prioritize being close to current, but not too far from dest
        # The destination term prevents building a tour that ends up far from the start
        score = 0.6 * d_current_norm + 0.4 * d_dest_norm
    else:
        # Do a 1-step lookahead: for each candidate, what's the best exit?
        # This is O(n^2) which is fine for n <= 15
        min_exit = np.full(n_unvisited, np.inf)
        for i in range(n_unvisited):
            if i == 0:
                continue
            # Distance from candidate i to all other unvisited
            # We want the minimum distance to another unvisited node (not itself)
            # Exclude the current node's candidate
            d_to_others = distance_matrix[unvisited_nodes[i], unvisited_nodes]
            # Set the distance to itself to inf
            d_to_others[i] = np.inf
            min_exit[i] = np.min(d_to_others)
        # For the first candidate, compute its exit
        d_to_others = distance_matrix[unvisited_nodes[0], unvisited_nodes]
        d_to_others[0] = np.inf
        min_exit[0] = np.min(d_to_others)
        
        min_exit_norm = min_exit / (np.max(min_exit) + 1e-10)
        
        # Score: close to current, close to dest, and has a good exit
        score = 0.5 * d_current_norm + 0.3 * d_dest_norm + 0.2 * min_exit_norm
    
    # Select the candidate with minimum score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
