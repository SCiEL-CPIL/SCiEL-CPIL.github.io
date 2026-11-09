"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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


def _soft_min(values, beta=5.0):
    """Smooth approximation of min via log-sum-exp (negative log-sum-exp)."""
    values = np.asarray(values, dtype=np.float64)
    # Clip very small values to avoid log(0)
    values = np.maximum(values, 1e-10)
    max_val = np.max(values)
    # soft min = -log(sum(exp(-beta * (v - min_ref)))) / beta
    # Use a stable form: soft_min ≈ min + (1/beta) * log(sum(exp(-beta*(v - min))))
    min_val = np.min(values)
    shifted = values - min_val
    # For numerical stability, only sum over values close to min
    # exp(-beta * shifted) where shifted >= 0
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return min_val + log_sum / beta


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # 1-step Lookahead Heuristic with smooth soft-min and destination awareness
    # For each candidate c:
    # 1. Cost to get to c from current: d(current, c)
    # 2. Cost to leave c:
    #    - Use soft-min over distances to other unvisited nodes (smooth surrogate)
    #    - Blend with distance to destination, weighted by urgency (1/n)
    #      As n decreases, destination becomes more relevant
    
    best_score = float('inf')
    best_node = candidates[0]
    
    # Pre-compute distances from current to all candidates
    dist_from_current = distance_matrix[current_node, candidates]
    
    # Destination urgency: as fewer nodes remain, the final closure to destination matters more
    # weight_dest increases from ~0.1 when n=50 to ~0.5 when n=2
    weight_dest = min(0.5, 0.1 + 0.4 * (1.0 - n / 50.0))
    
    # Soft-min temperature: higher beta = closer to hard min
    # Lower beta when more nodes remain (more options), higher beta when fewer
    beta = 3.0 + 2.0 * (1.0 - n / 50.0)
    
    for i, c in enumerate(candidates):
        # Cost to reach c
        cost_in = dist_from_current[i]
        
        if n == 2:
            # Only 2 nodes left: after picking c, we go to the other, then destination
            other_idx = 1 - i
            other_node = candidates[other_idx]
            cost_out = distance_matrix[c, other_node]
            # Add a small destination awareness term
            cost_dest = distance_matrix[c, destination_node]
            total_score = cost_in + cost_out + 0.3 * cost_dest
        else:
            # Distances from c to all other candidates (excluding self)
            dists_to_others = distance_matrix[c, candidates].copy()
            dists_to_others[i] = float('inf')
            
            # Smooth minimum over other unvisited nodes
            soft_min_cost = _soft_min(dists_to_others, beta=beta)
            
            # Destination distance for closure awareness
            cost_dest = distance_matrix[c, destination_node]
            
            # Blend: primarily soft-min to next node, with destination awareness
            cost_out = (1.0 - weight_dest) * soft_min_cost + weight_dest * cost_dest
        
        total_score = cost_in + cost_out
        
        if total_score < best_score:
            best_score = total_score
            best_node = c
            
    return int(best_node)
# EVOLVE-BLOCK-END
