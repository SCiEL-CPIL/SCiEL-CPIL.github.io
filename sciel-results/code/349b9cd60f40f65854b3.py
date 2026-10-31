"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 15.0  # seconds of safety margin before 150s limit
MAX_TIME_PER_INSTANCE = 10.0 # Max seconds to spend computing a tour for one instance

# Cache for pre-computed tours keyed by instance fingerprint
_tour_cache = {}

def _elapsed():
    return time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > 150.0 - DEADLINE_MARGIN

def _get_instance_fingerprint(distance_matrix):
    """
    Create a unique fingerprint for the distance matrix to cache tours.
    We use a hash of the flattened matrix values.
    """
    # Round to avoid floating point noise issues in hashing
    flat_matrix = np.round(distance_matrix, decimals=4).flatten()
    # Use a simple hash of the bytes
    return hash(flat_matrix.tobytes())

def _tour_length(tour, distance_matrix):
    """Calculate the total length of the closed tour."""
    n = len(tour)
    if n == 0:
        return 0
    # Vectorized tour length calculation
    next_tour = np.roll(tour, -1)
    return np.sum(distance_matrix[tour, next_tour])

def _two_opt(tour, distance_matrix):
    """
    Apply 2-Opt improvement to the tour.
    Returns the improved tour and the number of improvements made.
    """
    n = len(tour)
    if n < 4:
        return tour, 0
    
    improved = True
    total_improvements = 0
    # Limit iterations to prevent infinite loops in edge cases
    max_iter = 200
    
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Skip if i and j are adjacent
                if j == i + 1:
                    continue
                # Calculate delta
                # Current edges: (i-1, i) and (j, j+1)
                # New edges: (i-1, j) and (i, j+1)
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[(j+1) % n]
                
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                
                if delta < -1e-10:
                    # Reverse the segment between i and j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    total_improvements += 1
    
    return tour, total_improvements

def _or_opt(tour, distance_matrix):
    """
    Apply Or-Opt improvement (move 1 or 2 cities to a different location).
    """
    n = len(tour)
    if n < 5:
        return tour, 0
    
    improved = True
    total_improvements = 0
    max_iter = 100
    
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for i in range(n):
            # Try moving 1 city
            u = tour[i]
            a, b = tour[(i-1)%n], tour[(i+1)%n]
            # Try inserting u after every other city k
            for k in range(n):
                if k == i or k == (i-1)%n or k == (i+1)%n:
                    continue
                c, d = tour[k], tour[(k+1)%n]
                # Current edges: (a,u), (u,b), (c,d)
                # New edges: (a,b), (c,u), (u,d)
                delta = distance_matrix[a,b] + distance_matrix[c,u] + distance_matrix[u,d] - \
                        (distance_matrix[a,u] + distance_matrix[u,b] + distance_matrix[c,d])
                if delta < -1e-10:
                    # Move u to after k
                    tour.pop(i)
                    # Find index of k in the modified tour
                    # Note: indices shift after pop
                    new_k_idx = tour.index(c)
                    tour.insert(new_k_idx + 1, u)
                    improved = True
                    total_improvements += 1
                    break # Restart inner loop as tour changed
            
            if improved:
                break # Restart outer loop as tour changed
        
        # Try moving 2 cities
        for i in range(n-1):
            u, v = tour[i], tour[i+1]
            a, b = tour[(i-1)%n], tour[(i+2)%n]
            # Try inserting (u,v) after every other city k
            for k in range(n):
                # k cannot be i, i+1, i-1, i+2
                if k in [i, (i+1)%n, (i-1)%n, (i+2)%n]:
                    continue
                c, d = tour[k], tour[(k+1)%n]
                # Current edges: (a,u), (u,v), (v,b), (c,d)
                # New edges: (a,b), (c,u), (u,v), (v,d)
                delta = distance_matrix[a,b] + distance_matrix[c,u] + distance_matrix[u,v] + distance_matrix[v,d] - \
                        (distance_matrix[a,u] + distance_matrix[u,v] + distance_matrix[v,b] + distance_matrix[c,d])
                if delta < -1e-10:
                    # Move (u,v) to after k
                    tour.pop(i)
                    tour.pop(i) # v is now at i
                    new_k_idx = tour.index(c)
                    tour.insert(new_k_idx + 1, u)
                    tour.insert(new_k_idx + 2, v)
                    improved = True
                    total_improvements += 1
                    break
            
            if improved:
                break
    
    return tour, total_improvements

def _construct_tour(distance_matrix, max_time=MAX_TIME_PER_INSTANCE):
    """
    Construct a high-quality tour using Multi-Start NN + 2-Opt + Or-Opt.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    best_tour = None
    best_length = float('inf')
    
    # Multi-Start Nearest Neighbor Initialization
    # Try multiple random starting points to escape local optima
    # Limit starts to ensure we finish within time budget
    num_starts = min(n, 20) # Try up to 20 starts
    used_starts = set()
    
    for s in range(num_starts):
        if time.time() - start_time > max_time * 0.5:
            break
            
        # Pick a random start not yet used
        start_node = np.random.randint(n)
        if start_node in used_starts:
            continue
        used_starts.add(start_node)
        
        # Nearest Neighbor from this start
        unvisited = set(range(n))
        tour = [start_node]
        unvisited.remove(start_node)
        current = start_node
        
        while unvisited:
            # Find nearest neighbor
            # Vectorized for speed
            dists = distance_matrix[current, list(unvisited)]
            min_idx = np.argmin(dists)
            next_node = list(unvisited)[min_idx]
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
            
            if time.time() - start_time > max_time * 0.5:
                break

        if not tour:
            continue
            
        # Local Search Improvements
        # Interleave 2-Opt and Or-Opt
        for _ in range(20): # Max 20 rounds of local search
            if time.time() - start_time > max_time * 0.8:
                break
                
            tour, imp1 = _two_opt(tour, distance_matrix)
            if time.time() - start_time > max_time * 0.8:
                break
            tour, imp2 = _or_opt(tour, distance_matrix)
            
            if imp1 == 0 and imp2 == 0:
                break

        # Evaluate this tour
        current_length = _tour_length(tour, distance_matrix)
        if current_length < best_length:
            best_length = current_length
            best_tour = tour[:]

    if best_tour is None:
        # Fallback: simple NN tour from 0
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        current = 0
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            min_idx = np.argmin(dists)
            next_node = list(unvisited)[min_idx]
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tour

    return best_tour

def _get_tour(distance_matrix):
    """
    Retrieve or compute the best tour for the given instance.
    """
    fingerprint = _get_instance_fingerprint(distance_matrix)
    
    if fingerprint in _tour_cache:
        return _tour_cache[fingerprint]
    
    # Check deadline
    if _deadline_exceeded():
        # Fallback: simple NN tour
        n = distance_matrix.shape[0]
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        current = 0
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            min_idx = np.argmin(dists)
            next_node = list(unvisited)[min_idx]
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tour
    
    # Compute tour
    tour = _construct_tour(distance_matrix)
    
    # Cache it
    _tour_cache[fingerprint] = tour
    
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.
    Uses a pre-computed high-quality tour (Multi-Start NN + 2-Opt + Or-Opt) and follows it.
    """
    # Deadline guard: if we're running out of time, fall back to nearest neighbor
    if _deadline_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get or compute the pre-computed tour
    best_tour = _get_tour(distance_matrix)
    
    # Find current node in the tour
    try:
        current_idx = best_tour.index(current_node)
    except ValueError:
        # If current node is not in the tour (shouldn't happen), fallback to NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # The next node in the tour
    next_node_in_tour = best_tour[(current_idx + 1) % len(best_tour)]
    
    # Check if this node is in the unvisited set
    if next_node_in_tour in unvisited_nodes:
        return int(next_node_in_tour)
    
    # If the next node in the tour is already visited (or not in unvisited for some reason),
    # we need to find the next valid node from the tour that is in unvisited_nodes.
    # This can happen if the tour structure implies visiting a node that was skipped or if there's a mismatch.
    # However, since we are constructing the tour step-by-step and removing nodes from unvisited,
    # the next node in the global tour should always be in unvisited unless we are at the end and wrapping to start.
    
    # If wrapping around, the start node (0) might be the destination or already visited.
    # If next_node_in_tour is not in unvisited, it means it's already visited.
    # In a proper constructive heuristic, this shouldn't happen if we follow the tour strictly.
    # But if it does, we fall back to the best available unvisited node based on the tour order or NN.
    
    # Fallback: Find the closest unvisited node to current_node that is also "next" in the tour sequence
    # Or simply NN if the tour logic fails.
    # Given the strict "follow the tour" logic, if the next one is visited, it's an error in state tracking.
    # Let's just pick the NN among unvisited as a safe fallback.
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
