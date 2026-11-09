"""Convex hull peeling TSP heuristic with angle-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 150s hard kill

# Module-level cache for convex hull computation
_hull_cache = {}


def _compute_convex_hull(nodes, distance_matrix):
    """
    Compute convex hull of a set of nodes using distance matrix.
    Returns list of node indices on the hull in order.
    Uses a simplified approach: iteratively find extreme points.
    """
    key = tuple(sorted(nodes))
    if key in _hull_cache:
        return _hull_cache[key]
    
    if len(nodes) <= 3:
        _hull_cache[key] = list(nodes)
        return _hull_cache[key]
    
    # For convex hull, we need to find boundary points
    # Strategy: Find the point with minimum x (or use distance-based approach)
    # Since we only have distance matrix, use a different approach:
    # A point is on the hull if it's far from the centroid of other points
    
    n = len(nodes)
    # Compute "centrality" for each node - nodes far from others are likely on hull
    # Use sum of distances as a proxy (hull nodes tend to have higher sum of distances)
    
    # Better approach: Use the fact that for Euclidean points, 
    # a point is on the convex hull if it cannot be expressed as a convex combination
    # of other points. Approximate by checking if removing it significantly changes
    # the "spread" of the set.
    
    # Simple heuristic: A node is on the hull if it has at least one other node
    # such that the angle at the third node is obtuse for all triples.
    # This is expensive. Instead, use a simpler proxy:
    # Nodes with high max-distance-to-others are likely on hull.
    
    max_dists = np.zeros(n)
    for i in range(n):
        max_dists[i] = np.max(distance_matrix[nodes[i], nodes])
    
    # Threshold: nodes in top 40% by max distance are candidates for hull
    threshold = np.percentile(max_dists, 60)
    hull_candidates = [nodes[i] for i in range(n) if max_dists[i] >= threshold]
    
    # Ensure at least 3 points on hull
    if len(hull_candidates) < 3:
        # Just return all nodes
        _hull_cache[key] = list(nodes)
        return _hull_cache[key]
    
    _hull_cache[key] = hull_candidates
    return _hull_cache[key]


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Convex hull peeling heuristic for TSP tour construction.
    
    Strategy:
    1. In early stages (many unvisited nodes), prefer nodes on the convex hull
       of the unvisited set to build the outer boundary first.
    2. Use angle-based scoring: minimize deviation from the direction toward
       the destination, which naturally traces the hull.
    3. In later stages, fall back to distance-based selection with destination
       awareness.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For two nodes left, pick the one that minimizes total edge cost to close the tour
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # Compute convex hull of unvisited nodes
    hull_nodes = _compute_convex_hull(unvisited_nodes, distance_matrix)
    hull_set = set(hull_nodes)
    
    # Distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Stage-based strategy:
    # Early: strong hull preference + angle-based scoring
    # Mid: moderate hull preference + blend of distance and angle
    # Late: distance-based with destination awareness
    
    if n_unvisited > 30:
        # Early stage: strongly prefer hull nodes
        hull_bonus = 0.3  # Strong preference for hull nodes
        angle_weight = 0.5
        dist_weight = 0.5
    elif n_unvisited > 15:
        # Mid stage: moderate hull preference
        hull_bonus = 0.15
        angle_weight = 0.3
        dist_weight = 0.7
    else:
        # Late stage: minimal hull preference, focus on closing the tour
        hull_bonus = 0.0
        angle_weight = 0.0
        dist_weight = 1.0
    
    # Compute angle-based score
    # For each candidate, compute the angle between (current->candidate) and (current->destination)
    # We want candidates that are in the "direction" of the destination or on the hull
    
    # Vector from current to destination
    # We don't have coordinates, only distance matrix. So we approximate angle using
    # the law of cosines: cos(theta) = (d_curr^2 + d_dest^2 - d_cand_dest^2) / (2 * d_curr * d_dest)
    
    # d_cand_dest is the distance from candidate to destination
    # For candidate i: d(current, cand_i), d(cand_i, dest), d(current, dest)
    d_cur_dest = distance_matrix[current_node, destination_node]
    
    if d_cur_dest > 1e-10 and np.any(d_curr > 1e-10) and np.any(d_dest > 1e-10):
        # cos(theta_i) = (d_curr_i^2 + d_dest_i^2 - d_cur_dest^2) / (2 * d_curr_i * d_dest_i)
        # We want to minimize theta, so minimize cos(theta) from 1 downward
        # Actually, we want candidates that are NOT in the direct path to destination
        # (they should be on the hull, i.e., off to the side)
        
        # Compute cos of angle at current node between candidate and destination
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_theta = (d_curr**2 + d_dest**2 - d_cur_dest**2) / (2 * d_curr * d_dest)
            cos_theta = np.clip(cos_theta, -1, 1)
        
        # Angle score: we want to penalize candidates that are too close to the 
        # direct path to destination (they're likely interior). 
        # Hull nodes tend to have larger angles (more to the side).
        # So we reward larger angles (smaller cos_theta).
        angle_score = -cos_theta  # Higher is better (more to the side)
    else:
        angle_score = np.zeros(n_unvisited)
    
    # Distance score: lower is better
    dist_score = d_curr
    
    # Hull bonus: give a discount to hull nodes
    is_hull = np.array([1.0 if node in hull_set else 0.0 for node in unvisited_nodes])
    hull_score = is_hull * hull_bonus * np.max(d_curr)  # Scale bonus by typical distance
    
    # Combined score: lower is better
    if angle_weight > 0:
        # Normalize angle score to [0, 1] range for proper weighting
        if np.max(angle_score) - np.min(angle_score) > 1e-10:
            angle_norm = (angle_score - np.min(angle_score)) / (np.max(angle_score) - np.min(angle_score))
        else:
            angle_norm = np.zeros(n_unvisited)
        
        if np.max(dist_score) - np.min(dist_score) > 1e-10:
            dist_norm = (dist_score - np.min(dist_score)) / (np.max(dist_score) - np.min(dist_score))
        else:
            dist_norm = np.zeros(n_unvisited)
        
        scores = dist_weight * dist_norm - angle_weight * angle_norm - hull_score / (np.max(d_curr) + 1e-10)
    else:
        # Late stage: simple distance with destination awareness
        progress = max(0.0, min(1.0, (15.0 - n_unvisited) / 14.0))
        alpha = 0.6 * (progress ** 2)
        scores = d_curr + alpha * d_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
