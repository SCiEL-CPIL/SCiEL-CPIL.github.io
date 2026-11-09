"""Improved TSP constructive heuristic: direction-aware scoring with convex-hull bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for derived data to avoid recomputation across calls
_cache = {}
_deadline_margin = 5.0  # seconds to keep under 120s limit

def _get_cached_data(distance_matrix):
    """Cache derived data: coordinates (if square symmetric), hull info, etc."""
    key = id(distance_matrix)
    if key in _cache:
        return _cache[key]
    
    n = distance_matrix.shape[0]
    # Try to recover coordinates from distance matrix (for hull computation)
    # Use MDS-like approach or just work with distances directly
    # For simplicity and speed, we'll use a geometric heuristic based on distances
    
    # Compute "hull-ness" proxy: nodes with larger average distance to others are more likely on hull
    avg_dist = np.mean(distance_matrix, axis=1)
    max_dist = np.max(distance_matrix, axis=1)
    
    # Normalize
    avg_dist_norm = (avg_dist - avg_dist.min()) / (avg_dist.max() - avg_dist.min() + 1e-10)
    max_dist_norm = (max_dist - max_dist.min()) / (max_dist.max() - max_dist.min() + 1e-10)
    
    # Hull score: combination of average and max distance
    hull_score = 0.6 * avg_dist_norm + 0.4 * max_dist_norm
    
    _cache[key] = {
        'n': n,
        'hull_score': hull_score,
        'avg_dist': avg_dist,
    }
    return _cache[key]


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a multi-factor scoring:
    1. Distance to current node (primary)
    2. Direction continuity (angle penalty for sharp turns)
    3. Convex hull bias (prefer boundary nodes)
    4. Avoidance of destination until late (savings heuristic)
    """
    n = len(unvisited_nodes)
    if n == 0:
        return destination_node
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Get cached data
    cached = _get_cached_data(distance_matrix)
    hull_score = cached['hull_score']
    
    # Get distances from current to all unvisited
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from unvisited to destination (for savings/avoidance)
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Direction continuity: compute angle between (prev->current) and (current->candidate)
    # We need the previous node. Since we don't have full tour history, we approximate:
    # Use the fact that we're building a tour - the "incoming" direction can be estimated
    # from the current node's position relative to the centroid of unvisited nodes
    # For a simpler approach: penalize candidates that would create a long jump
    # relative to the local neighborhood
    
    # Compute a "locality" score: prefer nodes that are close to the current node's
    # local neighborhood (nodes that are also close to current)
    # This is a proxy for direction continuity
    
    # Angle-based scoring using distance matrix only:
    # If we went A -> B -> C, the "turn" is related to d(A,C) vs d(A,B) + d(B,C)
    # We don't know A, so we use a heuristic:
    # Penalize candidates where the path current->candidate->destination 
    # is much longer than direct current->destination (indicates detour)
    
    direct_to_dest = distance_matrix[current_node, destination_node]
    
    # Detour penalty: how much longer is going through candidate vs going direct
    # Only apply if we're not too early in the tour
    # Estimate tour progress: fewer unvisited means later in tour
    progress = 1.0 - n / cached['n']  # 0 at start, 1 near end
    
    # Score components:
    # 1. Primary: distance to current (normalized)
    dist_norm = dists / (dists.max() + 1e-10)
    
    # 2. Hull bonus: prefer hull nodes, weighted by progress (more important early)
    hull_bonus = hull_score[unvisited_nodes] * 0.3 * (1.0 - progress)
    
    # 3. Detour avoidance: penalize nodes that create long detours to destination
    #    Weight increases as we get closer to destination
    detour_cost = np.maximum(0, dists + dists_to_dest - direct_to_dest)
    detour_norm = detour_cost / (detour_cost.max() + 1e-10)
    detour_weight = 0.4 * progress  # Stronger penalty later in tour
    
    # 4. Local consistency: prefer nodes that are also close to other nearby nodes
    #    (avoids isolated jumps)
    # Compute for each candidate, the mean distance to the 3 nearest other unvisited
    if n > 3:
        # For efficiency, use a simple proxy: variance of distances from candidate to others
        # Lower variance = more central/consistent
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Mean distance from each candidate to all others (excluding self)
        mean_local = np.mean(sub_dist, axis=1)
        local_norm = mean_local / (mean_local.max() + 1e-10)
        local_weight = 0.15
    else:
        local_norm = np.zeros(n)
        local_weight = 0.0
    
    # Combined score (lower is better)
    score = (dist_norm * 1.0 
             - hull_bonus 
             + detour_norm * detour_weight 
             + local_norm * local_weight)
    
    # Handle NaN/inf
    score = np.where(np.isfinite(score), score, np.inf)
    
    # Select minimum score
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
