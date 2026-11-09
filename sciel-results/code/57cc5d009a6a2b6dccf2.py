"""Heuristic matrix for ant colony optimization of TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dm, start=0):
    """Generate a tour using the nearest neighbor heuristic."""
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dm[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour


def _tour_length(tour, dm):
    """Calculate the length of a closed tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total


def _two_opt(tour, dm, max_iter=2000):
    """Improve a tour using 2-opt local search with early termination."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
    return tour, best_len


def _or_opt(tour, dm):
    """Improve a tour using Or-opt (relocating single nodes)."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    max_passes = 100
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            node = tour[i]
            # Remove node at position i
            a = tour[i - 1]
            b = tour[(i + 1) % n]
            tour.pop(i)
            n_minus_1 = n - 1
            # Try inserting at each position
            best_delta = 0.0
            best_pos = -1
            for j in range(n_minus_1):
                if j == i or j == (i - 1) % n_minus_1:
                    continue
                c = tour[j - 1]
                d = tour[j]
                delta = (dm[c, node] + dm[node, d]) - (dm[c, d]) + (dm[a, b])
                # Wait, the original cost contributed dm[a,b] which is now removed
                # Original cost: ... a -> node -> b ...
                # New cost after removal: ... a -> b ...
                # Insertion cost: ... c -> node -> d ... replacing c -> d
                # So delta = (dm[c,node] + dm[node,d] - dm[c,d]) + dm[a,b]
                # This is correct.
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_pos = j
            if best_pos >= 0:
                tour.insert(best_pos, node)
                best_len += best_delta
                improved = True
            else:
                # Put it back
                tour.insert(i, node)
    return tour, best_len


def _multi_start_local_search(dm, start_time, deadline):
    """Run multiple starts with combined local search, respecting time budget."""
    n = dm.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Try a few different starting nodes for NN
    starts = [0, n // 2, n // 3, n // 4, (n * 3) // 4]
    
    for start_node in starts:
        if time.time() > deadline:
            break
            
        # Generate NN tour
        tour = _nearest_neighbor_tour(dm, start=start_node)
        
        # Apply 2-opt
        tour, length = _two_opt(tour, dm, max_iter=1000)
        
        # Apply Or-opt
        tour, length = _or_opt(tour, dm)
        
        # Apply 2-opt again
        tour, length = _two_opt(tour, dm, max_iter=1000)
        
        if length < best_len:
            best_len = length
            best_tour = tour[:]
            
        # Check time
        if time.time() > deadline:
            break
            
    return best_tour, best_len


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Uses a high-quality reference tour to boost heuristic values for those edges.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit (300s eval + buffer)
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # Base heuristic: inverse distance
    # Avoid division by zero
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Generate a high-quality reference tour using multi-start + 2-opt + Or-opt
    ref_tour, ref_len = _multi_start_local_search(dm, start_time, deadline)
    
    # Check time
    if time.time() > deadline or ref_tour is None:
        # Fallback to base heuristic if we ran out of time
        return np.maximum(base_heuristic, 1e-9)
    
    # Boost heuristic for edges in the reference tour
    # We want to make the ants prefer these edges.
    heuristic = base_heuristic.copy()
    
    # Mark edges in the reference tour
    # The tour is a cycle: 0 -> ... -> n-1 -> 0
    # Boost directed edges in the tour order
    boost_factor = 200.0
    
    for i in range(n):
        u = ref_tour[i]
        v = ref_tour[(i + 1) % n]
        heuristic[u, v] *= boost_factor
    
    # Ensure no values are not positive or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
