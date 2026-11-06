"""Alternative strategy: Batched Multi-Start Constructive + 2-Opt Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to cache the solution
_solution_tour = None
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin for 150s kill

def _reset_state():
    global _solution_tour, _start_time
    _solution_tour = None
    _start_time = None

def _complete_tour_nn(start_node, unvisited, dist_matrix):
    """
    Complete the tour from start_node visiting all unvisited nodes using nearest neighbor.
    Returns the list of nodes visited in order (excluding start_node).
    """
    remaining = unvisited.copy()
    path = []
    curr = start_node
    
    while len(remaining) > 0:
        if len(remaining) == 1:
            next_node = remaining[0]
            remaining = np.array([], dtype=int)
        else:
            dists = dist_matrix[curr, remaining]
            min_idx = np.argmin(dists)
            next_node = remaining[min_idx]
            remaining = remaining[remaining != next_node]
        
        path.append(next_node)
        curr = next_node
        
    return path

def _apply_2opt(tour, dist_matrix, max_passes=10):
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
                    # Restart loops to ensure we don't miss subsequent improvements
                    break
            if improved:
                break
                
    return t

def _solve_tsp(distance_matrix):
    """
    Solve the TSP for the given distance matrix using multi-start NN + 2-opt.
    Returns the best tour found as a list of nodes.
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return list(range(n))
    
    all_nodes = np.arange(n)
    best_tour = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = _TIME_LIMIT - 5.0 # Leave buffer
    
    # Multi-start: Try different starting nodes and random perturbations?
    # For N=50, NN is fast. Let's do a few random starts or just start from 0 and maybe a few others.
    # Actually, a common heuristic is to run NN from every node? That's 50 * O(N^2) = 50*2500 = 125k ops. Very fast.
    # Then 2-opt on each. 50 * 2-opt cost.
    # 2-opt is O(N^2) per pass. With N=50, it's cheap.
    
    # Strategy:
    # 1. Run NN from every node 0..N-1.
    # 2. Apply 2-opt to each resulting tour.
    # 3. Keep the best one.
    
    for start in range(n):
        # Check time
        if time.time() - start_time > max_time:
            break
            
        # Complete tour from start using NN
        # unvisited = all nodes except start
        unvisited = all_nodes[all_nodes != start]
        path = _complete_tour_nn(start, unvisited, distance_matrix)
        tour = [start] + path
        
        # Apply 2-opt
        tour = _apply_2opt(tour, distance_matrix, max_passes=5) # Limit passes for speed in loop
        
        # Calculate length
        # Using numpy for speed
        nodes = np.array(tour)
        total = np.sum(distance_matrix[nodes[:-1], nodes[1:]]) + distance_matrix[nodes[-1], nodes[0]]
        
        if total < best_len:
            best_len = total
            best_tour = tour
            
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _solution_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        
    # If we haven't solved the TSP yet, solve it now
    if _solution_tour is None:
        _solution_tour = _solve_tsp(distance_matrix)
        
    # If the solution is empty or invalid (shouldn't happen), fallback
    if not _solution_tour:
        # Fallback to simple nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # We have a full tour: _solution_tour is a permutation of all nodes.
    # We need to return the next node in the tour sequence after current_node.
    # The tour is a cycle.
    
    # Find the index of current_node in the tour
    # Since N is small (50), linear search is fine, or we can precompute a map.
    # Let's precompute a next_node map for efficiency? 
    # Or just find the index.
    
    try:
        idx = _solution_tour.index(current_node)
    except ValueError:
        # Should not happen if current_node is in the tour
        # Fallback
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    # The next node is at (idx + 1) % N
    next_node = _solution_tour[(idx + 1) % len(_solution_tour)]
    
    return int(next_node)

# EVOLVE-BLOCK-END
