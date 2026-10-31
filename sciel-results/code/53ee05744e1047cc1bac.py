"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for pre-computed structures to speed up repeated calls
# We'll use a simple cache keyed by the distance_matrix object id and size
_cache = {}

def _get_cache_key(distance_matrix, unvisited_nodes):
    """Create a cache key based on distance matrix and unvisited set."""
    # Use the id of the distance matrix and the tuple of unvisited nodes
    # But since unvisited changes each call, we cache based on distance_matrix only
    return id(distance_matrix)

def _precompute(distance_matrix):
    """Precompute useful structures from the distance matrix."""
    n = distance_matrix.shape[0]
    # For each node, get the k nearest neighbors (k=5 for diversity)
    k = min(5, n - 1)
    nn_indices = np.argsort(distance_matrix, axis=1)[:, :k]
    nn_dists = np.take_along_axis(distance_matrix, nn_indices, axis=1)
    # Also precompute the median distance for scaling
    med_dist = np.median(distance_matrix[distance_matrix > 0]) if n > 1 else 1.0
    return {
        'nn_indices': nn_indices,
        'nn_dists': nn_dists,
        'med_dist': med_dist,
        'n': n,
    }

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a physics-inspired force model combined with
    regret-based lookahead.
    
    Forces:
    - Attraction toward destination (guides tour closure)
    - Attraction toward current node (local continuity)
    - Repulsion from recently visited nodes (avoid clustering/crossings)
    
    Combined with a regret term: nodes that are "expensive" to insert later
    (high distance to their nearest unvisited neighbors) are prioritized.
    """
    if len(unvisited_nodes) == 0:
        return current_node  # Should not happen per contract
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check cache for precomputed structures
    cache_key = _get_cache_key(distance_matrix, unvisited_nodes)
    if cache_key not in _cache:
        _cache[cache_key] = _precompute(distance_matrix)
    precomp = _cache[cache_key]
    
    n = precomp['n']
    med_dist = precomp['med_dist']
    
    # Get distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited (for closure guidance)
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Regret term: how "expensive" is it to leave this node unvisited?
    # For each unvisited node, compute its min distance to other unvisited nodes
    # (excluding itself). This approximates insertion cost.
    # To be efficient, use precomputed nearest neighbors
    nn_indices = precomp['nn_indices']
    nn_dists = precomp['nn_dists']
    
    # For each unvisited node, find the minimum distance to another unvisited node
    # using the precomputed k-NN
    unvisited_set = set(unvisited_nodes.tolist())
    regret = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        # Check k nearest neighbors of this node
        neighbors = nn_indices[node]
        min_dist = np.inf
        for nb in neighbors:
            if nb in unvisited_set and nb != node:
                d = distance_matrix[node, nb]
                if d < min_dist:
                    min_dist = d
        if np.isinf(min_dist):
            # Fallback: min distance to any other unvisited
            mask = np.ones(n, dtype=bool)
            mask[node] = False
            mask[~np.isin(np.arange(n), unvisited_nodes)] = False
            if mask.sum() > 0:
                min_dist = np.min(distance_matrix[node, mask])
            else:
                min_dist = med_dist
        regret[i] = min_dist
    
    # Force-based scoring:
    # 1. Attraction to destination: lower dist_to_dest is better
    # 2. Attraction to current: lower dist_from_current is better
    # 3. Regret: higher regret means we should visit this node sooner
    
    # Normalize terms to comparable scales
    # Use a weighted combination that balances local (current) and global (dest) guidance
    
    # Weight the terms based on progress in the tour
    # Early in tour: weight current node more (build local structure)
    # Late in tour: weight destination more (prepare for closure)
    progress = 1.0 - len(unvisited_nodes) / max(1, n - 1)
    
    # Weights: interpolate from current-focused to destination-focused
    w_current = 0.7 - 0.4 * progress  # 0.7 early, 0.3 late
    w_dest = 0.3 + 0.4 * progress     # 0.3 early, 0.7 late
    w_regret = 0.3  # Constant regret weight
    
    # Score: lower is better
    # Normalize distances by median for scale invariance
    norm_dist_current = dist_from_current / (med_dist + 1e-10)
    norm_dist_dest = dist_to_dest / (med_dist + 1e-10)
    norm_regret = regret / (med_dist + 1e-10)
    
    score = (w_current * norm_dist_current + 
             w_dest * norm_dist_dest - 
             w_regret * norm_regret)
    
    # Select the node with the best (lowest) score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
