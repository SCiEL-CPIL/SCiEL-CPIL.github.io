"""TSP constructive heuristic: geometric alignment + MST-inspired residual cost."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 2 seconds before the 150s limit (safety margin)."""
    return _start_time + 147.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Combine geometric alignment (how well a candidate lies on the path
    from current to destination) with an MST-inspired residual cost estimate.
    
    The alignment score: d(current, cand) + d(cand, dest) - d(current, dest)
    measures how much longer the path goes if we route through cand vs. going
    directly. A value of 0 means cand is exactly on the straight line.
    
    The MST-inspired cost: for each candidate, estimate the additional cost to
    connect it into the growing tour, considering both its distance to current
    and an estimate of how it will connect to the remaining unvisited nodes.
    
    Combined score:
        score = w1 * alignment + w2 * d(current, cand) + w3 * mst_residual
    
    where weights adapt based on progress through the tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline check
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Extract relevant distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # Geometric alignment score: how much extra distance via candidate
    # Low (near 0) = candidate lies on the direct path (good for tour continuity)
    alignment = dists_from_current + dists_to_dest - dist_current_dest
    # Clamp to non-negative (triangle inequality ensures this, but numerical safety)
    alignment = np.maximum(alignment, 0.0)
    
    # Normalize alignment by the direct distance for scale invariance
    if dist_current_dest > 1e-10:
        alignment_norm = alignment / dist_current_dest
    else:
        alignment_norm = alignment
    
    # Estimate total nodes
    total_nodes = distance_matrix.shape[0]
    
    # Progress: 0 when all nodes unvisited, 1 when few remain
    if total_nodes > 1:
        progress = 1.0 - (n_unvisited / total_nodes)
    else:
        progress = 1.0
    
    # MST-inspired residual cost estimation:
    # For each candidate i, the cost to integrate it is roughly:
    #   d(current, i) + (average distance from i to other unvisited nodes) * factor
    # This approximates the incremental MST cost of adding node i.
    # 
    # We compute the average distance from each candidate to all other unvisited nodes
    # as a proxy for its "integration cost" into the residual structure.
    
    if n_unvisited > 2:
        # Compute pairwise distances among unvisited nodes (only the upper triangle sum)
        # dist_matrix_unvisited[i, j] = distance between unvisited[i] and unvisited[j]
        dist_matrix_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # Sum of distances from each node to all others (excluding self)
        row_sums = np.sum(dist_matrix_unvisited, axis=1) - np.diag(dist_matrix_unvisited)
        
        # Average distance from each candidate to other unvisited nodes
        avg_residual = row_sums / (n_unvisited - 1)
        
        # Normalize for scale invariance
        mean_avg_residual = np.mean(avg_residual)
        if mean_avg_residual > 1e-10:
            avg_residual_norm = avg_residual / mean_avg_residual
        else:
            avg_residual_norm = avg_residual
    else:
        avg_residual_norm = np.ones(n_unvisited)
    
    # Adaptive weights:
    # Early in tour: emphasize nearest-neighbor (local optimality)
    # Late in tour: emphasize alignment and residual cost (global structure)
    w_alignment = 0.3 + 0.5 * progress  # 0.3 to 0.8
    w_nearest = 0.7 - 0.3 * progress     # 0.7 to 0.4
    w_residual = 0.1 + 0.2 * progress    # 0.1 to 0.3
    
    # Normalize nearest-neighbor distance
    mean_dist_current = np.mean(dists_from_current)
    if mean_dist_current > 1e-10:
        dists_from_current_norm = dists_from_current / mean_dist_current
    else:
        dists_from_current_norm = dists_from_current
    
    # Combined score
    scores = (w_alignment * alignment_norm + 
              w_nearest * dists_from_current_norm + 
              w_residual * avg_residual_norm)
    
    # Deadline check
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
