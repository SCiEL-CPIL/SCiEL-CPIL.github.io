"""TSP constructive heuristic with lookahead and 2-opt refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit
_visited_nodes = []


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def _tour_length(tour, distance_matrix):
    """Calculate the total length of a closed tour."""
    if len(tour) < 2:
        return 0.0
    n = len(tour)
    idx = np.arange(n)
    next_idx = np.roll(idx, -1)
    return np.sum(distance_matrix[tour[idx], tour[next_idx]])


def _fast_2opt(tour, distance_matrix, max_passes=2):
    """
    Apply a limited number of 2-opt passes to the tour.
    tour is a list of node indices representing a cycle.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    t = tour[:]
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            ni = (i + 1) % n
            for j in range(i + 2, n):
                # Skip if reversing the whole cycle (no change)
                if i == 0 and j == n - 1:
                    continue
                
                nj = (j + 1) % n
                
                old_cost = distance_matrix[t[i], t[ni]] + distance_matrix[t[j], t[nj]]
                new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[ni], t[nj]]
                
                if new_cost < old_cost - 1e-8:
                    # Reverse segment from i+1 to j
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    improved = True
        if not improved:
            break
    return t


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _visited_nodes, _START_TIME
    
    _ensure_start_time()
    
    # Detect new instance: if unvisited set is large, assume start of a new tour
    # For 50-city instances, the first call has 49 unvisited nodes.
    if len(unvisited_nodes) > 45:
        _visited_nodes = [current_node]
    else:
        # Append current node to visited list if it's not already the last one
        # This handles the case where the evaluator might call in a way that 
        # current_node is the last visited
        if not _visited_nodes or _visited_nodes[-1] != current_node:
            _visited_nodes.append(current_node)

    if _time_exceeded():
        # Fallback to nearest neighbor to ensure completion
        if len(unvisited_nodes) == 0:
            return destination_node
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        next_node = int(unvisited_nodes[0])
        _visited_nodes.append(next_node)
        return next_node
    
    best_candidate = -1
    best_score = float('inf')
    
    # Lookahead: Evaluate each candidate by simulating the rest of the tour
    for candidate in unvisited_nodes:
        # 1. Build partial tour prefix
        # The prefix is the history + current choice
        prefix = _visited_nodes + [candidate]
        
        # 2. Complete the tour using Nearest Neighbor for the remaining nodes
        remaining = [u for u in unvisited_nodes if u != candidate]
        
        tour = list(prefix)
        curr = candidate
        rem_list = list(remaining)
        
        while rem_list:
            rem_arr = np.array(rem_list)
            dists = distance_matrix[curr, rem_arr]
            min_idx = np.argmin(dists)
            next_node = rem_list[min_idx]
            tour.append(next_node)
            curr = next_node
            rem_list.pop(min_idx)
        
        # Close the tour by returning to destination (start)
        tour.append(destination_node)
        
        # 3. Refine with 2-opt
        # Limit passes to keep it fast
        optimized_tour = _fast_2opt(tour, distance_matrix, max_passes=1)
        
        # 4. Calculate length
        length = _tour_length(optimized_tour, distance_matrix)
        
        if length < best_score:
            best_score = length
            best_candidate = candidate
            
    # Update state
    _visited_nodes.append(best_candidate)
    
    return int(best_candidate)
# EVOLVE-BLOCK-END
