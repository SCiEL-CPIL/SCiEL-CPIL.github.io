import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def _estimate_completion_cost(start, dest, rem, dm):
    """
    Estimate the cost of completing a tour from 'start' through all nodes in 'rem' to 'dest'.
    Uses a 2-step lookahead greedy strategy.
    """
    n = len(rem)
    if n == 0:
        return dm[start, dest]
    if n == 1:
        return dm[start, rem[0]] + dm[rem[0], dest]
    if n == 2:
        n0, n1 = rem[0], rem[1]
        cost0 = dm[start, n0] + dm[n0, n1] + dm[n1, dest]
        cost1 = dm[start, n1] + dm[n1, n0] + dm[n0, dest]
        return min(cost0, cost1)
    
    curr = start
    rem_list = list(rem)
    cost = 0.0
    
    while rem_list:
        m = len(rem_list)
        if m == 1:
            cost += dm[curr, rem_list[0]] + dm[rem_list[0], dest]
            break
        elif m == 2:
            n0, n1 = rem_list[0], rem_list[1]
            cost0 = dm[curr, n0] + dm[n0, n1] + dm[n1, dest]
            cost1 = dm[curr, n1] + dm[n1, n0] + dm[n0, dest]
            cost += min(cost0, cost1)
            break
        else:
            # 2-step lookahead
            # d_curr: distance from curr to all rem
            d_curr = dm[curr, rem_list]
            
            # rem_dist: distance matrix between all rem nodes
            rem_arr = np.array(rem_list)
            rem_dist = dm[np.ix_(rem_arr, rem_arr)]
            
            # We need min distance from each node in rem to any other node in rem
            # Set diagonal to inf so we don't pick self
            np.fill_diagonal(rem_dist, np.inf)
            min_next = np.min(rem_dist, axis=1)
            
            # Score for each candidate next node:
            # d(curr, next) + d(next, best_next_of_next)
            scores = d_curr + min_next
            
            best_idx = np.argmin(scores)
            next_node = rem_list[best_idx]
            cost += d_curr[best_idx]
            curr = next_node
            rem_list.pop(best_idx)
            
    return cost

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead greedy completion heuristic.
    Evaluates all unvisited nodes (or a large subset if time is tight) to find the one
    that leads to the lowest estimated total tour completion cost.
    """
    # Time guard
    remaining = _get_time_remaining()
    if remaining < 2.0:
        # Fallback to simple nearest neighbor
        if len(unvisited_nodes) == 0:
            return int(destination_node)
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        cands = unvisited_nodes
        scores = distance_matrix[current_node, cands] + distance_matrix[cands, destination_node]
        return int(cands[np.argmin(scores)])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])
    
    # For small n, do exact evaluation
    if n == 2:
        c0 = candidates[0]
        c1 = candidates[1]
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        return int(c0) if cost0 < cost1 else int(c1)
    
    best_candidate = -1
    best_score = float('inf')
    
    # Evaluate all candidates if time allows, otherwise top-k
    # For N=50, evaluating all is fast (~0.1s)
    # We will iterate through all candidates
    
    for c in candidates:
        rem = np.array([x for x in unvisited_nodes if x != c], dtype=int)
        total_cost = distance_matrix[current_node, c] + _estimate_completion_cost(c, destination_node, rem, distance_matrix)
        
        if total_cost < best_score:
            best_score = total_cost
            best_candidate = c
            
        # Check time periodically
        if len(candidates) > 10 and _get_time_remaining() < 5.0:
             # If time is running out, break and return best so far
             # But we need to ensure we have a valid best_candidate
             break

    return int(best_candidate)
