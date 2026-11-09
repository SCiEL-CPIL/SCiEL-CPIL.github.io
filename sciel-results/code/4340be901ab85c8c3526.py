"""Improved TSP constructive heuristic with 2-opt look-ahead, savings, and regret."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None
_DEADLINE_SECONDS = 135.0  # Safety margin before 150s limit


def _get_time_remaining():
    """Return seconds remaining before the deadline (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, _DEADLINE_SECONDS - elapsed)


def _two_opt_improve_segment(tour, dist_matrix, max_iters=3):
    """
    Apply limited 2-opt improvement to a small segment of the tour.
    tour: list of node indices
    dist_matrix: distance matrix
    Returns improved tour (list)
    """
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    
    for _ in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # Check if reversing segment i+1..j improves
                old_dist = dist_matrix[a, b] + dist_matrix[c, d]
                new_dist = dist_matrix[a, c] + dist_matrix[b, d]
                if new_dist < old_dist - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour.
    
    Strategy:
    1. If few nodes remain (<= 12), use 2-opt on the remaining sub-tour to find the optimal next step.
    2. If many nodes remain, use a weighted heuristic combining:
       - Nearest Neighbor distance
       - Savings (Insertion Cost): Cost of routing through candidate vs direct current->dest
       - Regret: Minimum distance from candidate to other unvisited nodes (avoids dead ends)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    # If only one node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If no nodes left (shouldn't happen), return current
    if n_unvisited == 0:
        return int(current_node)
    
    # Time check - if we're running out of time, fall back to nearest neighbor
    remaining_time = _get_time_remaining()
    if remaining_time < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Regime 1: Small remaining sets - use exact/semi-exact local search
    if n_unvisited <= 12:
        # Start with a reasonable initial order (Nearest Neighbor)
        remaining = list(unvisited_nodes)
        current_pos = current_node
        ordered = []
        while remaining:
            closest = min(remaining, key=lambda x: distance_matrix[current_pos, x])
            ordered.append(closest)
            remaining.remove(closest)
            current_pos = closest
        
        # Add destination at the end to form a closed-ish segment for evaluation
        # We want to pick the next node, so we look at the segment: current -> [unvisited] -> destination
        full_segment = [current_node] + ordered + [destination_node]
        
        # Apply 2-opt to improve this segment
        # We allow a few iterations. Since N is small, this is fast.
        improved_segment = _two_opt_improve_segment(full_segment, distance_matrix, max_iters=3)
        
        # The next node is the second element in the improved segment (index 1)
        # (Index 0 is current_node)
        if len(improved_segment) >= 2:
            return int(improved_segment[1])
        
        # Fallback if 2-opt failed or returned something weird
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Regime 2: Larger sets - use fast heuristic with Savings and Regret
    current_to_candidates = distance_matrix[current_node, unvisited_nodes]
    candidates_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # 1. Nearest Neighbor Score (normalized)
    # We normalize to [0, 1] for consistent weighting
    min_curr = np.min(current_to_candidates)
    max_curr = np.max(current_to_candidates)
    range_curr = max_curr - min_curr if max_curr > min_curr else 1.0
    norm_nn = (current_to_candidates - min_curr) / range_curr
    
    # 2. Savings / Insertion Cost
    # Cost of path current -> candidate -> destination vs current -> destination
    # savings = dist(current, cand) + dist(cand, dest) - dist(current, dest)
    # Lower savings is better (means the detour is small)
    insertion_cost = current_to_candidates + candidates_to_dest - dist_current_dest
    
    min_ins = np.min(insertion_cost)
    max_ins = np.max(insertion_cost)
    range_ins = max_ins - min_ins if max_ins > min_ins else 1.0
    norm_ins = (insertion_cost - min_ins) / range_ins
    
    # 3. Regret (Future Connectivity)
    # How expensive is the *next* step after picking this candidate?
    # Min distance from candidate to any *other* unvisited node.
    # If this is high, we are picking an isolated node, which is bad.
    # We compute this for all candidates at once using the sub-matrix.
    unvisited_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(unvisited_dist, np.inf)
    min_next_dist = np.min(unvisited_dist, axis=1)
    
    min_reg = np.min(min_next_dist)
    max_reg = np.max(min_next_dist)
    range_reg = max_reg - min_reg if max_reg > min_reg else 1.0
    norm_regret = (min_next_dist - min_reg) / range_reg
    
    # Weights:
    # NN is strong for local progress.
    # Insertion cost helps avoid bad detours towards the end.
    # Regret helps avoid getting stuck.
    # Tuning: NN 0.5, Insertion 0.3, Regret 0.2
    w_nn = 0.5
    w_ins = 0.3
    w_regret = 0.2
    
    scores = w_nn * norm_nn + w_ins * norm_ins + w_regret * norm_regret
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
