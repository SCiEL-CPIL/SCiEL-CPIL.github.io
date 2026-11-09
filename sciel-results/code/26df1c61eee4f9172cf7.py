"""Rewritten program for the traveling salesman constructive heuristic task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track time budget across calls
_start_time = None
_TIME_BUDGET = 140.0  # Safety margin before 150s limit


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.
    
    Uses a phase-aware blend:
    - Early phase: mostly nearest neighbor to build a compact tour
    - Late phase: blend in destination proximity to avoid a long closing edge
    
    The blend weight increases as fewer unvisited nodes remain, with a smooth
    transition. Also includes a mild look-ahead penalty to avoid creating
    isolated nodes that will be far from both current and destination.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = _get_elapsed()
    n_unvisited = len(unvisited_nodes)
    
    # If we're running out of time, fall back to fast nearest neighbor
    if elapsed > _TIME_BUDGET:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Determine phase based on how many nodes are left
    # Total nodes is typically 50, so we track progress
    # We don't know total directly, but n_unvisited tells us
    # Assume total ~ 50 for parameter tuning
    total_est = 50  # typical instance size
    
    # Progress: 0 at start (many unvisited), 1 at end (few unvisited)
    progress = 1.0 - (n_unvisited / float(total_est))
    progress = max(0.0, min(1.0, progress))
    
    # Blend weight: how much to weight destination proximity
    # Smooth sigmoid-like blend: starts near 0, ends near 1
    # Use a smoothstep-like function for gradual transition
    # alpha = 0 for first 60%, ramps to 1 by 100%
    if progress < 0.5:
        alpha = 0.0
    else:
        # Smooth ramp from 0.5 to 1.0
        t = (progress - 0.5) / 0.5
        alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Normalize distances to comparable scale
    max_d = max(np.max(dist_from_current), np.max(dist_to_dest), 1e-10)
    norm_current = dist_from_current / max_d
    norm_dest = dist_to_dest / max_d
    
    # Combined score: lower is better
    # Weighted blend of current-distance and destination-distance
    score = (1.0 - alpha) * norm_current + alpha * norm_dest
    
    # Add a mild look-ahead penalty: penalize nodes that are far from
    # the *average* of remaining unvisited nodes (to avoid stragglers)
    # This is a cheap proxy for future tour quality
    if n_unvisited > 2 and elapsed < _TIME_BUDGET - 5.0:
        # Mean position of unvisited nodes (in distance space, not coordinates)
        # Use the mean distance from current to all unvisited as a reference
        mean_dist = np.mean(dist_from_current)
        # Penalize nodes that are much farther than average from current
        # This discourages picking distant outliers early
        outlier_penalty = np.maximum(0, norm_current - mean_dist / max_d) * 0.1 * (1.0 - alpha)
        score = score + outlier_penalty
    
    # Argmin to pick best node
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
