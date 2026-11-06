"""Traveling salesman constructive heuristic with Regret-Based Selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_current_tour = []
_start_time = None
_TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = None

def _estimate_remaining_greedy(current_node, remaining_nodes, distance_matrix, start_node):
    """
    Fast greedy completion estimate from current_node through remaining_nodes back to start_node.
    Uses nearest-neighbor heuristic.
    """
    if len(remaining_nodes) == 0:
        return distance_matrix[current_node, start_node]
    
    remaining = list(remaining_nodes)
    curr = current_node
    total = 0.0
    
    while remaining:
        rem_arr = np.array(remaining)
        dists = distance_matrix[curr, rem_arr]
        min_idx = np.argmin(dists)
        total += dists[min_idx]
        curr = remaining[min_idx]
        remaining.pop(min_idx)
    
    # Close the tour
    total += distance_matrix[curr, start_node]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check - fallback to simple nearest neighbor
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    if n_unvisited == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        start_node = _current_tour[0]
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, start_node]
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, start_node]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _current_tour.append(next_node)
        return next_node
    
    # General case: n_unvisited >= 3
    # Regret-based selection: for each candidate, estimate total tour cost
    start_node = _current_tour[0]
    n_cands = n_unvisited
    
    # Precompute immediate costs
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, estimate the remaining cost
    est_costs = np.zeros(n_cands)
    
    # Optimization: for large n_cands, use a faster approximation
    # Instead of full greedy completion for each candidate, we can:
    # 1. For the first few candidates (top-k by immediate cost), do full estimation
    # 2. For others, use a simpler estimate
    # But for 50 nodes, full estimation should be fast enough
    
    for idx in range(n_cands):
        candidate = unvisited_nodes[idx]
        # Remaining nodes after picking candidate
        mask = np.ones(n_cands, dtype=bool)
        mask[idx] = False
        remaining = unvisited_nodes[mask]
        
        # Fast greedy completion
        est_costs[idx] = _estimate_remaining_greedy(candidate, remaining, distance_matrix, start_node)
    
    # Total estimated tour cost for each candidate
    total_costs = imm_costs + est_costs
    
    # Pick the candidate with minimum total estimated cost
    best_idx = np.argmin(total_costs)
    best_candidate = int(unvisited_nodes[best_idx])
    
    # Update global state
    _current_tour.append(best_candidate)
    
    return best_candidate

# EVOLVE-BLOCK-END
