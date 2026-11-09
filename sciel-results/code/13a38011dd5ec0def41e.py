"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # 120s limit, keep 5s margin

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a combined heuristic that favors:
    1. Convex hull-like behavior (angle-based to avoid crossings)
    2. Distance to nearest unvisited
    3. Distance to destination (to close the tour)
    
    The idea is to trace around the convex hull of unvisited points first,
    then fill in interior points, reducing edge crossings.
    """
    if len(unvisited_nodes) == 0:
        return current_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to nearest neighbor if out of time
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute distances from current to all unvisited
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from each unvisited to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # If only a few nodes left, use a simple combined score
    if len(unvisited_nodes) <= 5:
        # Weighted combination: prefer nodes close to current and close to destination
        score = 0.7 * dists + 0.3 * dists_to_dest
        return int(unvisited_nodes[np.argmin(score)])
    
    # Compute angle-based score to encourage convex hull tracing
    # Get coordinates indirectly through distance matrix? No, we don't have coords.
    # But we can use the triangle inequality and distances to estimate angles.
    
    # Alternative approach: use a "savings" heuristic similar to Clarke-Wright
    # For each candidate next node j, the savings of going current->j->dest 
    # vs current->dest + j->dest
    # But we don't know the full tour structure yet.
    
    # Let's try a different geometric approach using the distance matrix.
    # Compute for each unvisited node j:
    # - d(current, j): distance from current
    # - d(j, dest): distance to destination  
    # - d(current, dest): distance from current to destination
    
    # The angle at current between the vector to j and the vector to dest
    # can be estimated using the law of cosines:
    # cos(angle) = (d(current,j)^2 + d(current,dest)^2 - d(j,dest)^2) / (2*d(current,j)*d(current,dest))
    
    d_cd = distance_matrix[current_node, destination_node]
    if d_cd < 1e-10:
        # Current is destination, just pick nearest
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Compute cosines of angles
    # cos_theta = (d_cj^2 + d_cd^2 - d_jd^2) / (2 * d_cj * d_cd)
    d_cj = dists
    d_jd = dists_to_dest
    
    # Avoid division by zero
    denom = 2.0 * d_cj * d_cd
    denom_safe = np.where(denom < 1e-10, 1e-10, denom)
    
    cos_theta = (d_cj**2 + d_cd**2 - d_jd**2) / denom_safe
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # We want to prefer nodes that are in the "forward" direction relative to destination
    # i.e., small angle means we're heading toward destination
    # But for convex hull tracing, we actually want to go around the hull,
    # which means we want nodes that are NOT directly toward destination but 
    # rather to the side, creating a hull-like path.
    
    # Actually, let's think differently. The convex hull heuristic works by:
    # - Always going to the next point that maintains the hull boundary
    # - This means we should prefer points that are "extreme" in some direction
    
    # A simpler effective heuristic: combine nearest-neighbor with a penalty
    # for nodes that would create a "sharp turn" away from the general direction
    # toward the destination.
    
    # Let's use a weighted score:
    # score = alpha * d_cj + beta * d_jd + gamma * (1 - cos_theta)
    # where gamma encourages going in the direction of destination
    
    alpha = 0.5
    beta = 0.3
    gamma = 0.2
    
    # (1 - cos_theta) is 0 when going directly toward dest, 2 when going away
    # We want to minimize this, so small angle (toward dest) is good
    angle_penalty = 1.0 - cos_theta
    
    score = alpha * d_cj + beta * d_jd + gamma * angle_penalty
    
    # Normalize scores for stability
    score_min = np.min(score)
    score_max = np.max(score)
    if score_max - score_min > 1e-10:
        score_norm = (score - score_min) / (score_max - score_min)
    else:
        score_norm = score
    
    return int(unvisited_nodes[np.argmin(score_norm)])
# EVOLVE-BLOCK-END
