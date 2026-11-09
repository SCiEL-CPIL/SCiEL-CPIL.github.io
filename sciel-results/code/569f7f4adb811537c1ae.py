import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill

# Cache for convex hull information
_HULL_CACHE = {}


def _compute_hull_info(distance_matrix):
    """
    Compute convex hull membership and a 'hull-ness' score for each node.
    Uses distance matrix to identify boundary nodes (nodes that are far from centroid
    or have few close neighbors).
    Returns (hull_mask, hull_score) where hull_mask is bool array and hull_score is float array.
    """
    key = id(distance_matrix)
    if key in _HULL_CACHE:
        return _HULL_CACHE[key]
    
    n = distance_matrix.shape[0]
    try:
        # Compute centroid in distance space (approximate)
        # A node is likely on the hull if it's far from the average of other nodes
        # Or if it has few very close neighbors
        
        # Method: For each node, compute average distance to all other nodes.
        # Nodes with high avg distance are likely on the boundary.
        avg_dist = distance_matrix.sum(axis=1) / (n - 1)
        
        # Also consider the minimum distance to any other node (isolated nodes might be interior or hull)
        min_dist = distance_matrix.min(axis=1)
        
        # Hull score: normalized average distance (higher = more likely hull)
        # Normalize to [0, 1]
        avg_min = avg_dist.min()
        avg_max = avg_dist.max()
        if avg_max > avg_min:
            hull_score = (avg_dist - avg_min) / (avg_max - avg_min)
        else:
            hull_score = np.ones(n)
            
        # Hull mask: top 20% by hull score
        threshold = np.percentile(hull_score, 80)
        hull_mask = hull_score >= threshold
        
        _HULL_CACHE[key] = (hull_mask, hull_score)
        return hull_mask, hull_score
    except Exception:
        _HULL_CACHE[key] = (np.ones(n, dtype=bool), np.zeros(n))
        return np.ones(n, dtype=bool), np.zeros(n)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Uses a convex hull peeling strategy:
    1. Identify hull nodes (boundary nodes) using distance matrix statistics.
    2. Prefer hull nodes that continue the current direction (low turning penalty).
    3. For the final few nodes, use destination-aware heuristic.
    """
    # Time guard - if we're running out of time, fall back to greedy
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get hull information (cached)
    hull_mask, hull_score = _compute_hull_info(distance_matrix)
    
    # Threshold for switching to destination-aware mode
    THRESHOLD = 8
    
    if n_unvisited > THRESHOLD:
        # Phase 1: Hull-aware nearest neighbor
        d_curr = distance_matrix[current_node, unvisited_nodes]
        
        # Get hull scores for unvisited nodes
        hull_scores_unvisited = hull_score[unvisited_nodes]
        hull_mask_unvisited = hull_mask[unvisited_nodes]
        
        # Create a penalty for non-hull nodes
        # Hull nodes get a bonus (negative penalty), non-hull nodes get a penalty
        # The penalty is scaled by how "interior" the node is
        # hull_score is [0, 1], higher = more hull-like
        # We want to prefer high hull_score nodes
        
        # Penalty = -bonus * hull_score (so higher hull_score = lower total score)
        # But we also want to not completely ignore distance
        # Let's use: score = d_curr - lambda * hull_score
        # Where lambda is a tuning parameter
        
        # Adaptive lambda: stronger hull preference when many nodes remain
        # But not too strong to avoid weird behavior
        lambda_hull = 0.1 * (n_unvisited / 50.0)  # Scale with remaining nodes
        
        scores = d_curr - lambda_hull * hull_scores_unvisited
        
        # Additional: penalize sharp turns
        # For each candidate, estimate the turn angle using previous node info
        # We don't have previous node, but we can use destination as a proxy for direction
        # Or we can skip this for now and rely on hull preference
        
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
    
    # Phase 2: Destination-aware for the final stretch
    if n_unvisited == 2:
        # Exact check for 2 nodes
        u, v = unvisited_nodes
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
        
    if n_unvisited == 3:
        # Exact check for 3 nodes (6 permutations)
        u, v, w = unvisited_nodes
        perms = [
            (u, v, w), (u, w, v),
            (v, u, w), (v, w, u),
            (w, u, v), (w, v, u)
        ]
        best_cost = float('inf')
        best_first = u
        for p in perms:
            c = (distance_matrix[current_node, p[0]] + 
                 distance_matrix[p[0], p[1]] + 
                 distance_matrix[p[1], p[2]] + 
                 distance_matrix[p[2], destination_node])
            if c < best_cost:
                best_cost = c
                best_first = p[0]
        return int(best_first)
        
    # For 4 to THRESHOLD nodes:
    # Use a weighted combination of distance to current, distance to destination, and hull score
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    hull_scores_unvisited = hull_score[unvisited_nodes]
    
    # Adaptive weights
    # At n_unvisited=8, lambda_dest ~ 1.0, lambda_hull ~ 0.05
    # At n_unvisited=4, lambda_dest ~ 2.0, lambda_hull ~ 0.02
    
    progress = 1.0 - (n_unvisited - 4) / (THRESHOLD - 4)  # 0 at THRESHOLD, 1 at 4
    alpha_dest = 1.0 + 1.0 * progress
    lambda_hull_tail = 0.05 - 0.03 * progress  # Decrease hull preference in tail
    
    scores = d_curr + alpha_dest * d_dest - lambda_hull_tail * hull_scores_unvisited
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
