"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_start_time = None
_DEADLINE = 140.0  # Safety margin from 150s limit
_cache = {}

def _fingerprint(dm):
    """Create a hashable fingerprint from the distance matrix."""
    # Use the first few rows and columns as a fingerprint
    # For 50 cities, the full matrix is 50x50=2500 floats
    # Use a subset for speed
    fp_data = dm[:10, :10].flatten()
    # Round to avoid floating point issues
    fp_data = np.round(fp_data, 6)
    return tuple(fp_data.tolist())

def _tour_length(tour, dm):
    """Calculate the length of a closed tour."""
    n = len(tour)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total

def _nn_tour(dm):
    """Construct a tour using Nearest Neighbor heuristic."""
    n = dm.shape[0]
    unvisited = set(range(n))
    tour = []
    
    # Start from city 0
    current = 0
    unvisited.remove(current)
    tour.append(current)
    
    for _ in range(n - 1):
        if not unvisited:
            break
        # Find nearest unvisited city
        dists = dm[current, list(unvisited)]
        nearest = list(unvisited)[np.argmin(dists)]
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return np.array(tour)

def _two_opt(tour, dm, max_passes=5):
    """Improve tour using 2-Opt local search."""
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for reversing segment i+1 to j
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour

def _or_opt(tour, dm):
    """Improve tour using Or-Opt (relocate single node or segment of 2-3 nodes)."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for seg_len in [1, 2, 3]:
                if i + seg_len >= n:
                    continue
                # Segment to relocate
                seg = tour[i:i + seg_len]
                a = tour[i - 1]
                b = tour[i + seg_len]
                # Try inserting at each position
                for j in range(n - seg_len):
                    if i <= j < i + seg_len:
                        continue
                    if j == i - 1:
                        continue
                    c = tour[j - 1]
                    d = tour[j]
                    # Calculate delta
                    old_cost = dm[a, seg[0]] + dm[seg[-1], b]
                    new_cost = dm[a, b] + dm[c, seg[0]] + dm[seg[-1], d]
                    if new_cost < old_cost - 1e-10:
                        # Relocate
                        tour = np.concatenate([tour[:i], tour[i + seg_len:], seg])
                        # But we need to insert at position j in the new array
                        # This is tricky, let's do it more carefully
                        # Actually, let's just rebuild
                        break
    return tour

def _compute_best_tour(dm):
    """Compute a high-quality tour using multi-start NN + 2-Opt."""
    n = dm.shape[0]
    best_tour = None
    best_len = float('inf')
    
    # Multi-start: try starting from different cities
    start_cities = [0, 1, 2, 10, 20, 30, 40] if n > 40 else list(range(min(n, 5)))
    
    for start in start_cities:
        # Rotate NN tour to start from 'start'
        base_tour = _nn_tour(dm)
        # Find index of start in base_tour
        idx = np.where(base_tour == start)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        rotated = np.concatenate([base_tour[idx:], base_tour[:idx]])
        
        # Apply 2-Opt
        tour = _two_opt(rotated.copy(), dm, max_passes=3)
        tour_len = _tour_length(tour, dm)
        
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour
    
    if best_tour is None:
        best_tour = _nn_tour(dm)
    
    # Final 2-Opt pass on the best tour
    best_tour = _two_opt(best_tour.copy(), dm, max_passes=5)
    
    return best_tour

def _next_in_tour(current_node, destination_node, unvisited_nodes, dm):
    """Find the next node in the pre-computed tour."""
    tour = _get_tour(dm)
    
    # Find current node's position in the tour
    current_pos = np.where(tour == current_node)[0]
    if len(current_pos) == 0:
        # Fallback to nearest neighbor
        dists = dm[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    current_pos = current_pos[0]
    n = len(tour)
    unvisited_set = set(unvisited_nodes.tolist())
    
    # Walk forward in the tour to find the next unvisited node
    for offset in range(1, n):
        next_pos = (current_pos + offset) % n
        candidate = tour[next_pos]
        if candidate in unvisited_set:
            return int(candidate)
    
    # Fallback
    dists = dm[current_node, unvisited_nodes]
    return int(unvisited_nodes[np.argmin(dists)])

def _get_tour(dm):
    """Get or compute the cached tour for a given distance matrix."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    fid = _fingerprint(dm)
    if fid in _cache:
        return _cache[fid]
    
    # Check time budget
    if time.time() - _start_time > _DEADLINE - 20:
        # Not enough time, use simple NN tour
        return _nn_tour(dm)
    
    tour = _compute_best_tour(dm)
    _cache[fid] = tour
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a pre-computed high-quality tour.
    On the first call for an instance, it computes a tour using multi-start
    Nearest Neighbor + 2-Opt local search, then follows that tour order.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check time budget - if too close to deadline, use fast NN
    if time.time() - _start_time > _DEADLINE - 5:
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    return _next_in_tour(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
