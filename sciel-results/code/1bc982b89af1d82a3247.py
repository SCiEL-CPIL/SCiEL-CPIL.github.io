"""Rewritten TSP constructive heuristic with fast lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = 115.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _time_exceeded():
    return _elapsed() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Picks the next city in a TSP tour.
    
    Strategy:
    - Fast 2-step lookahead heuristic.
    - If time is running out, fall back to nearest neighbor.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is running out, use simple nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # 2-step lookahead
    # For each candidate A in unvisited:
    # Cost = dist(current, A) + min(dist(A, B) for B in unvisited \ {A})
    # If unvisited has only 1 node, Cost = dist(current, A) + dist(A, destination)
    
    best_score = np.inf
    best_next = unvisited_nodes[0]
    
    # Convert to list for easier manipulation if needed, but numpy is fine
    # We need to exclude A when finding the next nearest neighbor
    
    # Optimization: Pre-compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    for i, candidate in enumerate(unvisited_nodes):
        # Cost to reach candidate
        cost = dist_from_current[i]
        
        if n_unvisited > 1:
            # Find the nearest neighbor of candidate among the other unvisited nodes
            # Create a mask to exclude the candidate itself
            # We can just slice or use a boolean mask
            # Since n is small (< 50), creating a sub-array is fast
            other_indices = np.concatenate((unvisited_nodes[:i], unvisited_nodes[i+1:]))
            if len(other_indices) > 0:
                dists_to_others = distance_matrix[candidate, other_indices]
                min_next_dist = np.min(dists_to_others)
                cost += min_next_dist
            else:
                # Should not happen if n_unvisited > 1
                pass
        else:
            # Last node, must return to destination
            cost += distance_matrix[candidate, destination_node]
        
        if cost < best_score:
            best_score = cost
            best_next = candidate
            
    return int(best_next)
# EVOLVE-BLOCK-END
