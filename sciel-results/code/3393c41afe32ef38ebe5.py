"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_DEADLINE_SECONDS = 115.0  # safety margin under 120s limit
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _check_deadline():
    return _get_elapsed() > _DEADLINE_SECONDS


def _greedy_completion_cost(current, dest, unvisited, dist_matrix):
    """
    Simulate greedy nearest-neighbor completion from current through unvisited
    to dest, returning total path length. This captures the actual trajectory
    of the constructive heuristic rather than abstract distance weights.
    """
    total = 0.0
    remaining = list(unvisited)
    cur = current
    
    while remaining:
        # Find nearest unvisited to current
        idx = np.argmin(dist_matrix[cur, remaining])
        next_node = remaining[idx]
        total += dist_matrix[cur, next_node]
        remaining.pop(idx)
        cur = next_node
    
    # Final jump to destination
    total += dist_matrix[cur, dest]
    return total


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using 2-step lookahead:
    For each candidate, estimate the total completion cost if we pick that
    candidate next and then continue greedily. Pick the candidate that
    minimizes this lookahead cost.
    
    This structural change replaces abstract distance weighting with
    simulation of the actual greedy trajectory, better capturing how
    early choices affect the full tour length.
    """
    # Deadline guard: if we're over budget, use fastest fallback
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # For each candidate, simulate greedy completion and score by total cost
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    for i, candidate in enumerate(unvisited_nodes):
        # Cost to reach candidate from current
        cost_to_candidate = distance_matrix[current_node, candidate]
        
        # Remaining nodes after picking this candidate
        remaining = np.delete(unvisited_nodes, i)
        
        # Simulate greedy completion from candidate through remaining to destination
        # Use a lightweight version: just sum distances in greedy order
        # For efficiency, we do a quick greedy simulation
        rem_list = list(remaining)
        cur = candidate
        sim_cost = 0.0
        
        while rem_list:
            idx = np.argmin(distance_matrix[cur, rem_list])
            next_node = rem_list[idx]
            sim_cost += distance_matrix[cur, next_node]
            rem_list.pop(idx)
            cur = next_node
        
        sim_cost += distance_matrix[cur, destination_node]
        total_cost = cost_to_candidate + sim_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = candidate
    
    return int(best_node)
# EVOLVE-BLOCK-END
