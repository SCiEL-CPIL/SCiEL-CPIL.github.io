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
    
    if n == 2:
        # Exact calculation for 2 remaining nodes
        a, b = candidates[0], candidates[1]
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return int(a if cost_a <= cost_b else b)
    
    if n == 3:
        # Exact calculation for 3 remaining nodes
        a, b, c = candidates
        # Try all 6 permutations
        import itertools
        best_cost = float('inf')
        best_first = a
        for perm in itertools.permutations([a, b, c]):
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            if cost < best_cost:
                best_cost = cost
                best_first = perm[0]
        return int(best_first)
    
    # For n >= 4: use a 2-step lookahead with angular penalty
    # Score for candidate c:
    #   primary: d(current, c) + min_{v != c} d(c, v)
    #   secondary: penalty for "bad angles" - if the best next node from c is far from the overall direction to destination
    #   Also consider: how many close neighbors does c have? (flexibility)
    
    # Pre-compute distances
    dist_from_current = distance_matrix[current_node, candidates]
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    # For each candidate, compute:
    # 1. Immediate cost: d(current, c)
    # 2. Best continuation: min_{v != c} d(c, v)
    # 3. Flexibility: second-best continuation (to avoid dead ends)
    # 4. Destination alignment: how well c's neighborhood aligns with destination
    
    scores = np.zeros(n)
    
    for i, c in enumerate(candidates):
        # Immediate cost
        cost_in = dist_from_current[i]
        
        # Get distances from c to all other candidates
        dist_c_to_others = distance_matrix[c, candidates]
        dist_c_to_others[i] = np.inf  # exclude self
        
        # Best and second-best continuation
        sorted_dists = np.sort(dist_c_to_others)
        best_cont = sorted_dists[0]
        second_best_cont = sorted_dists[1] if n > 2 else 0.0
        
        # Flexibility penalty: if best and second-best are very different, c is a "trap"
        # (you're forced to go to a specific next node)
        flexibility_penalty = 0.3 * max(0, second_best_cont - best_cont)
        
        # Destination alignment: compute the angle between (c - current) and (dest - c)
        # If they point in roughly the same direction, it's good
        # Use dot product as a proxy for alignment
        # In 2D: we don't have coordinates, so use distance-based proxy
        # d(current, dest) vs d(current, c) + d(c, dest)
        # If d(current, c) + d(c, dest) is close to d(current, dest), c is "on the way"
        d_current_dest = distance_matrix[current_node, destination_node]
        d_current_c_plus_c_dest = dist_from_current[i] + dist_to_dest[i]
        # Alignment score: lower is better (closer to direct path)
        alignment_penalty = 0.2 * max(0, d_current_c_plus_c_dest - d_current_dest)
        
        # Total score
        scores[i] = cost_in + best_cont + flexibility_penalty + alignment_penalty
    
    best_idx = np.argmin(scores)
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
