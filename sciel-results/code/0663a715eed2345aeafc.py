"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_time_limit = 140.0  # Safety margin for 150s kill
_instance_id = -1

def _reset_state():
    global _current_tour, _start_time, _instance_id
    _current_tour = []
    _start_time = time.time()
    _instance_id += 1

def _fast_2opt(tour, distance_matrix, max_passes=1):
    """
    Fast single-pass 2-opt for a closed tour.
    Returns improved tour as list of ints.
    """
    n = len(tour)
    if n < 4:
        return list(tour)
    
    t = list(tour)
    
    for _pass in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if j == n - 1:
                    a, b, c, d = t[i], t[i+1], t[j], t[0]
                else:
                    a, b, c, d = t[i], t[i+1], t[j], t[j+1]
                old = distance_matrix[a, b] + distance_matrix[c, d]
                new = distance_matrix[a, c] + distance_matrix[b, d]
                if new < old - 1e-10:
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    improved = True
        if not improved:
            break
    return t

def _tour_length(tour, distance_matrix):
    """Compute closed tour length."""
    n = len(tour)
    if n <= 1:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        total += distance_matrix[tour[i], tour[i+1]]
    total += distance_matrix[tour[-1], tour[0]]
    return total

def _nn_completion(start_node, remaining_nodes, distance_matrix):
    """Complete tour from start_node visiting all remaining_nodes via NN. Returns list of visited order."""
    if len(remaining_nodes) == 0:
        return []
    result = []
    curr = start_node
    rem = list(remaining_nodes)
    while rem:
        dists = distance_matrix[curr, rem]
        idx = np.argmin(dists)
        next_node = rem[idx]
        result.append(next_node)
        curr = next_node
        rem.pop(idx)
    return result

def _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by evaluating each candidate with NN completion + fast 2-opt.
    Adapts effort based on time remaining and number of nodes left.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    elapsed = time.time() - _start_time
    time_left = _time_limit - elapsed
    
    # Determine how many 2-opt passes to use based on time budget
    # Each candidate requires O(n_unvisited) for NN + O(n_total^2) for 2-opt
    # Total work ≈ n_unvisited * (n_unvisited + n_total^2 * passes)
    # For n_total=50, n_total^2 = 2500
    # Budget: use 2-opt only if we have enough time
    n_total = len(_current_tour) + n_unvisited
    
    # Estimate cost per candidate: NN is O(n_unvisited), 2-opt is O(n_total^2 * passes)
    # We have time_left seconds. Each 2-opt pass on n=50 is roughly 2500 operations.
    # Be conservative: allow 2-opt if time_left > 2.0 * n_unvisited * n_total^2 / 1e6
    # Simplify: allow 2-opt if time_left > 0.5 and n_unvisited < 30
    use_2opt = time_left > 1.0 and n_unvisited <= 40
    
    best_candidate = -1
    best_score = float('inf')
    
    # Pre-compute distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Sort candidates by distance from current to break ties and potentially prune
    # But we still need to evaluate all for correctness of lookahead
    for idx in range(n_unvisited):
        candidate = unvisited_nodes[idx]
        
        # Build simulated tour
        sim_tour = list(_current_tour) + [candidate]
        
        # Remaining nodes
        remaining = list(unvisited_nodes)
        remaining.pop(idx)
        
        # NN completion
        completed = _nn_completion(candidate, remaining, distance_matrix)
        sim_tour.extend(completed)
        
        # Evaluate
        if use_2opt:
            # Check time again
            if time.time() - _start_time > _time_limit - 0.5:
                score = _tour_length(sim_tour, distance_matrix)
            else:
                optimized = _fast_2opt(sim_tour, distance_matrix, max_passes=1)
                score = _tour_length(optimized, distance_matrix)
        else:
            score = _tour_length(sim_tour, distance_matrix)
        
        if score < best_score:
            best_score = score
            best_candidate = candidate
    
    return int(best_candidate)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time, _instance_id
    
    # Detect new instance: if unvisited_nodes size is close to full (>= 45 for 50-city),
    # or if this is the very first call, reset state
    n_total_possible = distance_matrix.shape[0]
    if _start_time is None or len(unvisited_nodes) >= n_total_possible - 1:
        _reset_state()
        _current_tour = [current_node]
    
    # Time check - hard fallback
    elapsed = time.time() - _start_time
    if elapsed > _time_limit:
        if len(unvisited_nodes) == 0:
            return -1
        # Nearest neighbor fallback
        if len(unvisited_nodes) == 1:
            next_node = int(unvisited_nodes[0])
        else:
            next_node = int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        _current_tour.append(next_node)
        return next_node
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # Use lookahead strategy
    next_node = _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Update global state
    _current_tour.append(next_node)
    
    return next_node

# EVOLVE-BLOCK-END
