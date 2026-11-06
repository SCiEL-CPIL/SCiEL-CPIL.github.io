"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, fall back to nearest neighbor for safety
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, use nearest neighbor (fast and good enough)
    if n_unvisited <= 3:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from each unvisited to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute local density: for each unvisited node, the minimum distance to other unvisited nodes
    # This helps prefer nodes that are well-connected to remaining nodes
    # Using a soft-min approach to smooth the min operation
    if n_unvisited <= 30:
        # Compute pairwise distances between unvisited nodes
        pairwise = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # For each node, find the nearest neighbor among unvisited (excluding itself)
        # Set diagonal to inf so self-distance is ignored
        np.fill_diagonal(pairwise, np.inf)
        # Use soft-min (log-sum-exp trick) for smoothness
        # soft_min(x, tau) = -tau * log(sum(exp(-x/tau)))
        tau = 0.1
        soft_min = -tau * np.log(np.sum(np.exp(-pairwise / tau), axis=1))
        # Normalize soft_min to [0, 1] range for weighting
        sm_min = soft_min.min()
        sm_max = soft_min.max()
        if sm_max > sm_min:
            density_score = (soft_min - sm_min) / (sm_max - sm_min)
        else:
            density_score = np.zeros_like(soft_min)
    else:
        # For larger sets, approximate with distance to centroid of unvisited
        centroid = distance_matrix[unvisited_nodes, :].mean(axis=0)
        # This is expensive, so use a simpler approximation:
        # Use the mean distance from each unvisited to all unvisited
        mean_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].mean(axis=1)
        md_min = mean_dist.min()
        md_max = mean_dist.max()
        if md_max > md_min:
            density_score = (mean_dist - md_min) / (md_max - md_min)
        else:
            density_score = np.zeros_like(mean_dist)
    
    # Adaptive alpha: when few nodes remain, weight destination more heavily
    if n_unvisited <= 5:
        alpha = 0.5
        beta = 0.3  # weight for density
    elif n_unvisited <= 10:
        alpha = 0.3
        beta = 0.2
    else:
        alpha = 0.1
        beta = 0.15
    
    # Combined score: minimize distance from current, avoid bad endings, prefer well-connected nodes
    scores = dist_from_current + alpha * dist_to_dest + beta * density_score
    
    # Find the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
