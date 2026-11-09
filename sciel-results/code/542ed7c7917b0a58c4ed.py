"""TSP constructive heuristic with 2-step lookahead and global connectivity awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def _greedy_completion_cost(start_node, remaining_nodes, destination_node, distance_matrix):
    """
    Estimate the cost of completing a tour from start_node through remaining_nodes
    to destination_node using greedy nearest-neighbor.
    
    This simulates the actual trajectory rather than abstract distance weighting.
    """
    if len(remaining_nodes) == 0:
        return float(distance_matrix[start_node, destination_node])
    
    if len(remaining_nodes) == 1:
        return float(distance_matrix[start_node, remaining_nodes[0]] + distance_matrix[remaining_nodes[0], destination_node])
    
    total_cost = 0.0
    current = start_node
    unvisited = list(remaining_nodes)
    
    while unvisited:
        # Find nearest unvisited node
        dists = distance_matrix[current, unvisited]
        nearest_idx = np.argmin(dists)
        nearest_node = unvisited[nearest_idx]
        total_cost += dists[nearest_idx]
        current = nearest_node
        unvisited.pop(nearest_idx)
    
    total_cost += float(distance_matrix[current, destination_node])
    return total_cost


def _two_step_lookahead_score(current_node, candidate, remaining_after, destination_node, distance_matrix):
    """
    Compute a 2-step lookahead score for a candidate.
    
    Score = cost(current -> candidate) + estimated_completion_cost(candidate, remaining_after, destination)
    
    This captures how early choices affect full tour completion.
    """
    d1 = float(distance_matrix[current_node, candidate])
    
    if len(remaining_after) == 0:
        return d1 + float(distance_matrix[candidate, destination_node])
    
    if len(remaining_after) == 1:
        return d1 + float(distance_matrix[candidate, remaining_after[0]] + distance_matrix[remaining_after[0], destination_node])
    
    # For larger remaining sets, use a cheaper approximation:
    # d1 + MST-like lower bound + adjustment
    # Use the minimum spanning tree weight of remaining_after ∪ {candidate, destination}
    # as a proxy for the path cost
    
    # Quick approximation: sum of two nearest neighbors for each node in remaining_after
    # plus connection costs
    rem_list = list(remaining_after)
    n_rem = len(rem_list)
    
    # For efficiency with larger sets, use a simplified estimate
    if n_rem > 20:
        # Use average-based heuristic
        avg_dist = np.mean(distance_matrix[rem_list, rem_list])
        # Estimate path cost as roughly (n_rem+1) * avg_dist * 0.8 (heuristic factor)
        est_path_cost = (n_rem + 1) * avg_dist * 0.8
        return d1 + est_path_cost
    
    # For smaller sets, do actual greedy simulation
    return d1 + _greedy_completion_cost(candidate, remaining_after, destination_node, distance_matrix)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using 2-step lookahead heuristic.
    
    For each candidate, we estimate the total tour cost if we pick that candidate next,
    then complete the rest greedily. This captures how early node choices affect
    the full tour completion cost.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if close, fall back to simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For very small instances, evaluate all candidates with full lookahead
    if n_unvisited <= 20:
        scores = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            cand = unvisited_nodes[i]
            # Remaining after picking candidate
            remaining_after = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
            scores[i] = _two_step_lookahead_score(current_node, cand, remaining_after, destination_node, distance_matrix)
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
    
    else:
        # For larger instances, use a reduced candidate set
        # Get top-k nearest neighbors from current node
        k = min(n_unvisited, 12)
        dists = distance_matrix[current_node, unvisited_nodes]
        top_k_indices = np.argsort(dists)[:k]
        candidates = unvisited_nodes[top_k_indices]
        
        # Also include node closest to destination for connectivity
        dest_dists = distance_matrix[destination_node, unvisited_nodes]
        closest_to_dest = unvisited_nodes[np.argmin(dest_dists)]
        if closest_to_dest not in candidates:
            candidates = np.append(candidates, closest_to_dest)
        
        # Evaluate candidates with lookahead
        scores = np.zeros(len(candidates))
        for i in range(len(candidates)):
            cand = candidates[i]
            # Find remaining after picking candidate
            mask = unvisited_nodes != cand
            remaining_after = unvisited_nodes[mask]
            scores[i] = _two_step_lookahead_score(current_node, cand, remaining_after, destination_node, distance_matrix)
        
        best_idx = np.argmin(scores)
        return int(candidates[best_idx])
# EVOLVE-BLOCK-END
