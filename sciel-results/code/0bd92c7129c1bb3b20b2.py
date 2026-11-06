"""Alternative strategy: Fast vectorized look-ahead with greedy completion and light local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin for 150s kill

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _complete_tour_greedy(start_node, unvisited, dist_matrix):
    """
    Complete the tour from start_node visiting all unvisited nodes using nearest neighbor.
    Returns the list of nodes visited in order (excluding start_node which is already in current_tour).
    """
    remaining = unvisited.copy()
    path = []
    curr = start_node
    
    while len(remaining) > 0:
        # Find nearest neighbor in remaining
        if len(remaining) == 1:
            next_node = remaining[0]
            remaining = np.array([], dtype=int)
        else:
            dists = dist_matrix[curr, remaining]
            min_idx = np.argmin(dists)
            next_node = remaining[min_idx]
            # Remove next_node from remaining
            # Using boolean mask or delete. For small N, delete is fine, but mask is safer for type.
            # Since remaining is a copy, we can just filter it out.
            remaining = remaining[remaining != next_node]
        
        path.append(next_node)
        curr = next_node
        
    return path

def _apply_2opt(tour, dist_matrix, max_passes=1):
    """
    Apply 2-opt improvement to a closed tour (list of nodes).
    The tour is considered a cycle: 0 -> 1 -> ... -> N-1 -> 0.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    t = tour[:]
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Check if reversing t[i+1:j+1] improves the tour
                # Edges removed: (t[i], t[i+1]) and (t[j], t[j+1] if j < n-1 else t[j], t[0])
                # Edges added:   (t[i], t[j]) and (t[i+1], t[j+1] if j < n-1 else t[i+1], t[0])
                
                if j == n - 1:
                    # j is the last node, so the edge is (t[n-1], t[0])
                    d1 = dist_matrix[t[i], t[i+1]]
                    d2 = dist_matrix[t[j], t[0]]
                    d3 = dist_matrix[t[i], t[j]]
                    d4 = dist_matrix[t[i+1], t[0]]
                else:
                    d1 = dist_matrix[t[i], t[i+1]]
                    d2 = dist_matrix[t[j], t[j+1]]
                    d3 = dist_matrix[t[i], t[j]]
                    d4 = dist_matrix[t[i+1], t[j+1]]
                
                if d3 + d4 < d1 + d2 - 1e-8:
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    improved = True
                    # Restart loops to ensure we don't miss subsequent improvements in the same pass
                    # But for speed, we can just continue. However, restarting is standard for convergence.
                    # Given the small N=50, restarting is cheap enough for 1-2 passes.
                    break
            if improved:
                break
                
    return t

def _evaluate_tour(tour, dist_matrix, use_2opt=True):
    """
    Calculate the length of the closed tour.
    tour: list of nodes representing the full path from start to end.
    The evaluator closes the tour by adding edge from last node to start node.
    So we sum edges (t[i], t[i+1]) for i=0..N-2 and (t[N-1], t[0]).
    """
    if use_2opt:
        tour = _apply_2opt(tour, dist_matrix, max_passes=1)
        
    n = len(tour)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
        
    # Sum distances
    # Using numpy for speed
    # Create array of nodes
    nodes = np.array(tour)
    # Edges: (0,1), (1,2), ..., (N-2, N-1), (N-1, 0)
    # dist_matrix[nodes[:-1], nodes[1:]] gives (N-1) distances
    # dist_matrix[nodes[-1], nodes[0]] gives 1 distance
    total = np.sum(dist_matrix[nodes[:-1], nodes[1:]]) + dist_matrix[nodes[-1], nodes[0]]
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    # If only one node left, must pick it.
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node

    # Strategy: Look-ahead with Greedy Completion + 2-opt
    # For each candidate in unvisited_nodes:
    # 1. Add candidate to current tour.
    # 2. Complete the rest of the tour using Nearest Neighbor from candidate.
    # 3. Evaluate the resulting closed tour length (with 1 pass of 2-opt for refinement).
    # 4. Pick the candidate with the lowest estimated tour length.
    
    best_candidate = -1
    best_score = float('inf')
    
    # To save time, we can limit the number of candidates we deeply evaluate if time is tight.
    # But N=50 is small. Evaluating all ~50 candidates is feasible.
    
    # Pre-compute remaining nodes for each candidate?
    # unvisited_nodes is an array.
    
    for candidate in unvisited_nodes:
        # Create a new tour state: current_tour + [candidate]
        # Simulate the rest from 'candidate'
        
        # Get remaining nodes excluding candidate
        mask = unvisited_nodes != candidate
        remaining_nodes = unvisited_nodes[mask]
        
        # Complete the tour using NN
        # Start from candidate
        path = _complete_tour_greedy(candidate, remaining_nodes, distance_matrix)
        
        # Full tour: _current_tour + [candidate] + path
        # Note: _current_tour already includes current_node at the end?
        # Yes, _current_tour is the list of visited nodes so far, ending with current_node.
        # So the new tour is _current_tour + [candidate] + path.
        
        sim_tour = _current_tour + [candidate] + path
        
        # Check time again
        if time.time() - _start_time > _TIME_LIMIT - 5:
             # Fallback: just use distance to candidate
             score = distance_matrix[current_node, candidate]
        else:
             # Evaluate with 1 pass of 2-opt
             score = _evaluate_tour(sim_tour, distance_matrix, use_2opt=True)
        
        if score < best_score:
            best_score = score
            best_candidate = candidate
            
    # Update global state
    if best_candidate != -1:
        _current_tour.append(best_candidate)
        return int(best_candidate)
    else:
        # Fallback
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

# EVOLVE-BLOCK-END
