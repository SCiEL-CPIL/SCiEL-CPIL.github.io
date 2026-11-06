"""Traveling salesman constructive heuristic with Regret-2 selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 110.0  # Safety margin for 120s/150s limit

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    # If only one node left, must pick it.
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # If two nodes left, pick the one that gives shorter closing
    if len(unvisited_nodes) == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        start_node = _current_tour[0]
        # Option 1: a then b, then back to start
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, start_node]
        # Option 2: b then a, then back to start
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, start_node]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _current_tour.append(next_node)
        return next_node
    
    # Vectorized Regret-2 Heuristic
    n_cands = len(unvisited_nodes)
    
    # 1. Immediate costs from current_node to all candidates
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Compute Regret-2 for all candidates
    # We need the distance matrix between unvisited nodes
    # sub_matrix[i, j] = d(unvisited[i], unvisited[j])
    sub_matrix = distance_matrix[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
    
    # Mask the diagonal (distance to self) with infinity
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Find the smallest and second smallest distance for each node
    # np.argpartition is faster than full sort for k-th smallest
    # We need the 2 smallest, so k=2
    # indices of the 2 smallest
    indices = np.argpartition(sub_matrix, 2, axis=1)[:, :2]
    
    # Extract the values
    rows = np.arange(n_cands)
    d_1 = sub_matrix[rows, indices[:, 0]]
    d_2 = sub_matrix[rows, indices[:, 1]]
    
    # Regret is the difference
    regrets = d_2 - d_1
    
    # 3. Combine Immediate Cost and Regret
    # We want to minimize the total estimated impact.
    # High regret means we should visit this node soon.
    # So we add a penalty to the cost if regret is high?
    # Or we subtract regret from cost?
    # If we use Score = imm_cost + alpha * regret, we are penalizing nodes with high regret.
    # This is WRONG. We want to PICK nodes with high regret.
    # So we should MINIMIZE (imm_cost - alpha * regret) or MINIMIZE (imm_cost) but BREAK TIES by max regret.
    # Or, better: Score = imm_cost + alpha * (max_regret - regret)? No.
    
    # Standard approach:
    # If we don't pick u, we might incur a cost related to regret.
    # Let's try: Score = imm_cost + alpha * regret.
    # Wait, if regret is high, it's "bad" to delay.
    # So we want to pick high regret.
    # So we want to MINIMIZE (imm_cost - alpha * regret).
    
    # Let's test alpha = 1.0
    # If regret is 10 and imm_cost is 5, score = -5.
    # If regret is 2 and imm_cost is 6, score = 4.
    # We pick the first one.
    
    alpha = 1.0
    scores = imm_costs - alpha * regrets
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    best_candidate = int(unvisited_nodes[best_idx])
    
    # Update global state
    _current_tour.append(best_candidate)
    
    return best_candidate

# EVOLVE-BLOCK-END
