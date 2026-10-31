"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache to avoid recomputing across calls within the same instance
_cache = {}
_DEADLINE_MARGIN = 10.0  # seconds of safety margin
_START_TIME = None


def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + 150.0 - _DEADLINE_MARGIN


def _compute_savings(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    For each candidate j in unvisited_nodes, compute the savings of going
    current -> j -> destination versus current -> destination directly.
    savings(j) = d(current, destination) - d(current, j) - d(j, destination)
    We want to maximize savings (equivalent to minimizing the extra cost).
    """
    d_cd = distance_matrix[current_node, destination_node]
    d_cj = distance_matrix[current_node, unvisited_nodes]
    d_jd = distance_matrix[unvisited_nodes, destination_node]
    # Extra cost of inserting j between current and destination
    extra_cost = d_cj + d_jd
    # We want to minimize extra_cost, but also consider that we're replacing d_cd
    # So the net change is extra_cost - d_cd. Minimizing net change = minimizing extra_cost
    # But actually we want to maximize savings = d_cd - extra_cost
    # So minimizing extra_cost is equivalent
    return extra_cost


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a look-ahead heuristic that considers
    both the immediate distance and the resulting connection to the destination.
    """
    global _cache
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    deadline = _get_deadline()
    
    # Use a combined heuristic: 
    # score(j) = d(current, j) + alpha * d(j, destination)
    # This balances immediate proximity with future connectivity.
    # alpha = 0.5 is a reasonable default for Euclidean TSP.
    
    d_cj = distance_matrix[current_node, unvisited_nodes]
    d_jd = distance_matrix[unvisited_nodes, destination_node]
    
    # Primary heuristic: minimize d(current,j) + 0.5 * d(j,destination)
    # This is a simple but effective look-ahead
    alpha = 0.5
    scores = d_cj + alpha * d_jd
    
    # Check if we have time for a more sophisticated approach
    elapsed = time.time() - _START_TIME if _START_TIME is not None else 0
    if elapsed < 120.0:
        # Try a 2-step look-ahead for the top candidates
        top_k = min(10, n_unvisited)
        top_indices = np.argsort(scores)[:top_k]
        top_candidates = unvisited_nodes[top_indices]
        
        # For each top candidate, estimate the cost of the next step
        # after choosing this candidate
        best_score = np.inf
        best_candidate = top_candidates[0]
        
        for k_idx in range(len(top_candidates)):
            cand = top_candidates[k_idx]
            # Cost so far: d(current, cand)
            cost_so_far = distance_matrix[current_node, cand]
            
            # Estimate future: from cand, what's the best next move?
            # Approximate by looking at remaining unvisited nodes
            remaining_mask = np.ones(n_unvisited, dtype=bool)
            # Find index of cand in unvisited_nodes
            cand_pos = np.where(unvisited_nodes == cand)[0]
            if len(cand_pos) > 0:
                remaining_mask[cand_pos[0]] = False
            
            if np.sum(remaining_mask) > 0:
                remaining_nodes = unvisited_nodes[remaining_mask]
                # Approximate next cost: min over remaining of d(cand, r) + 0.5 * d(r, dest)
                d_cr = distance_matrix[cand, remaining_nodes]
                d_rd = distance_matrix[remaining_nodes, destination_node]
                next_est = np.min(d_cr + alpha * d_rd)
            else:
                next_est = distance_matrix[cand, destination_node]
            
            total_est = cost_so_far + next_est
            if total_est < best_score:
                best_score = total_est
                best_candidate = cand
        
        return int(best_candidate)
    else:
        # Fallback to simple heuristic
        return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
