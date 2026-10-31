"""
TSP constructive heuristic: Pre-compute a high-quality reference tour via
multi-start nearest-neighbor + 2-opt, then guide the constructive phase by
following the reference tour's successor while incorporating a small
connectivity penalty to avoid isolating distant nodes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_start_time = time.time()
_LIMIT = 135.0  # Safety margin before 150s limit

# Per-instance state
_current_instance_id = -1
_reference_successor = None  # array: successor[i] = next node after i in reference tour
_reference_tour = None
_instance_counter = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start):
    """Nearest-neighbor tour construction from a given start node."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def _two_opt(tour, distance_matrix):
    """Apply 2-opt improvement to a tour until no improvement is found."""
    n = len(tour)
    if n <= 3:
        return tour
    
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += distance_matrix[t[i], t[(i + 1) % n]]
        return total
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if _is_deadline_passed():
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Check if reversing tour[i+1..j] improves the tour
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # Current edges: a-b and c-d
                # New edges: a-c and b-d
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
        if _is_deadline_passed():
            break
    return tour


def _build_reference_tour(distance_matrix):
    """Build a high-quality reference tour using multi-start NN + 2-opt."""
    global _reference_successor, _reference_tour
    
    n = distance_matrix.shape[0]
    
    # Try multiple starts (at least 10, up to n if time permits)
    best_tour = None
    best_len = float('inf')
    
    # Use all starts for n <= 50, but limit total time
    num_starts = min(n, 50)
    starts = list(range(num_starts))
    
    for start in starts:
        if _is_deadline_passed():
            break
        tour = _nn_tour(distance_matrix, start)
        tour = _two_opt(tour, distance_matrix)
        # Compute tour length
        total = 0.0
        for i in range(n):
            total += distance_matrix[tour[i], tour[(i + 1) % n]]
        if total < best_len:
            best_len = total
            best_tour = tour[:]
    
    if best_tour is None:
        # Fallback: simple NN from node 0
        best_tour = _nn_tour(distance_matrix, 0)
    
    _reference_tour = best_tour
    
    # Build successor array: successor[i] = the node that comes after i in the tour
    _reference_successor = np.full(n, -1, dtype=int)
    for i in range(n):
        _reference_successor[best_tour[i]] = best_tour[(i + 1) % n]
    
    return best_tour


def _exact_tail_optimization(current, unvisited, distance_matrix, destination):
    """For small remaining sets, try to find the best completion."""
    if len(unvisited) <= 8:
        # Brute force or near-brute force
        from itertools import permutations
        best_order = None
        best_len = float('inf')
        for perm in permutations(unvisited):
            total = distance_matrix[current, perm[0]]
            for i in range(len(perm) - 1):
                total += distance_matrix[perm[i], perm[i + 1]]
            total += distance_matrix[perm[-1], destination]
            if total < best_len:
                best_len = total
                best_order = perm
        if best_order is not None:
            return int(best_order[0])
    return None


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy:
    1. On first call for an instance, pre-compute a reference tour.
    2. Guide selection by preferring the successor in the reference tour,
       with a small penalty for nodes that are far from remaining nodes.
    3. For small remaining sets, use exact tail optimization.
    """
    global _current_instance_id, _reference_successor, _reference_tour, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)
    
    if is_new_instance:
        _instance_counter += 1
        _reference_successor = None
        _reference_tour = None
        _build_reference_tour(distance_matrix)
    
    if _is_deadline_passed():
        # Fallback: nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = int(np.argmin(dists))
        return int(unvisited_nodes[best_idx])
    
    # For small remaining sets, try exact optimization
    if n_unvisited <= 8:
        result = _exact_tail_optimization(current_node, unvisited_nodes, distance_matrix, destination_node)
        if result is not None:
            return result
    
    # Score each candidate
    scores = np.zeros(n_unvisited)
    
    # Base: distance from current
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores += dists_from_current
    
    # Strong pull towards the reference tour successor
    if _reference_successor is not None:
        # For each unvisited node, check if it's the successor of current in the reference tour
        # If so, give it a bonus
        ref_successor_of_current = _reference_successor[current_node]
        if ref_successor_of_current in unvisited_nodes:
            idx = list(unvisited_nodes).index(ref_successor_of_current)
            scores[idx] -= 0.5 * dists_from_current[idx]  # Strong preference
    
    # Connectivity penalty: avoid isolating nodes that are far from all remaining
    if n_unvisited > 2:
        # For each unvisited node, compute its min distance to other unvisited nodes
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_dist_to_others = np.min(sub_dist, axis=1)
        # Penalize nodes that are far from others (they might get isolated)
        # Use a soft penalty
        avg_min_dist = np.mean(min_dist_to_others)
        scores += 0.1 * (min_dist_to_others - avg_min_dist)
    
    # Small pull towards destination to avoid long closing edge
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    scores += 0.15 * dists_to_dest
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
