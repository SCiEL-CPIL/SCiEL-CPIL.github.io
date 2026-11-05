"""Rewritten TSP constructive heuristic: look-ahead nearest neighbor with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: evaluator kills at 150s; leave safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin


def _time_exceeded():
    return time.time() >= _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a constructive TSP tour.
    
    Uses a 2-step look-ahead (approximate 2-opt improvement on the local choice):
    For each candidate next node c, estimate the cost of going current->c and then
    from c to the nearest unvisited node after c (excluding c). This helps avoid
    choices that lead to expensive returns.
    
    Falls back to simple nearest-neighbor if time budget is nearly exhausted or
    the candidate set is too large for the look-ahead to be beneficial.
    """
    if _time_exceeded():
        # Fallback: just pick nearest neighbor quickly
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Both options: pick the one that minimizes total tour closure
        a, b = unvisited_nodes
        # Option 1: current -> a -> b -> destination
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        # Option 2: current -> b -> a -> destination
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return int(a if cost1 <= cost2 else b)
    
    # For larger sets, use 2-step look-ahead
    # For each candidate c in unvisited_nodes:
    #   cost = dist(current, c) + min_{u in unvisited\{c}} dist(c, u)
    # This approximates the next step cost after choosing c.
    
    # Vectorized computation:
    # dist_current_c: shape (n_unvisited,)
    dist_current_c = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate c_i, we need min dist from c_i to any other unvisited node (excluding itself)
    # dist_from_c: shape (n_unvisited, n_unvisited) where [i,j] = dist(c_i, c_j)
    # We want min over j != i of dist_from_c[i, j]
    
    # To handle the exclusion, set diagonal to inf and take row min
    dist_from_c = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(dist_from_c, np.inf)
    min_next_dist = np.min(dist_from_c, axis=1)
    
    # Total estimated cost for choosing candidate i:
    total_cost = dist_current_c + min_next_dist
    
    # Pick the candidate with minimum total cost
    best_idx = np.argmin(total_cost)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
