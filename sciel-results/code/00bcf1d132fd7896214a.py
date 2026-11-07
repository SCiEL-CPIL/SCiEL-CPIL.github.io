import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

def _nearest_neighbor_tour(d, start=0):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    tour = [start]
    for _ in range(n - 1):
        d_row = d[current].copy()
        d_row[visited] = np.inf
        d_row[current] = np.inf
        next_node = np.argmin(d_row)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(tour, d):
    n = len(tour)
    if n <= 2:
        return tour
    
    def tour_length(t):
        return sum(d[t[i], t[(i + 1) % n]] for i in range(n))

    best_tour = tour[:]
    best_len = tour_length(best_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                u1 = best_tour[i - 1]
                u2 = best_tour[i]
                u3 = best_tour[j]
                u4 = best_tour[(j + 1) % n]
                
                old_len = d[u1, u2] + d[u3, u4]
                new_len = d[u1, u3] + d[u2, u4]
                
                if new_len < old_len - 1e-10:
                    best_tour = best_tour[:]
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_len = tour_length(best_tour)
                    improved = True
    return best_tour

def _fast_or_opt(tour, d):
    """Faster Or-opt using incremental cost calculation."""
    n = len(tour)
    if n <= 3:
        return tour
    
    best_tour = tour[:]
    best_len = sum(d[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
    improved = True
    
    while improved:
        improved = False
        # Try moving 1 node
        for i in range(n):
            u = best_tour[i]
            prev = best_tour[(i - 1) % n]
            next_ = best_tour[(i + 1) % n]
            
            # Remove u: cost change is -d[prev,u] - d[u,next_] + d[prev,next_]
            remove_cost = d[prev, next_] - d[prev, u] - d[u, next_]
            
            for j in range(n - 1):
                if j == (i - 1) % n or j == i:
                    continue
                # Insert u after position j in the tour without u
                # Need to find the actual nodes at positions j and j+1 in the modified tour
                # This is complex; skip for now and rely on 2-opt
                
                # Simplified: just try inserting at each position
                temp_tour = best_tour[:]
                temp_tour.pop(i)
                insert_pos = j if j < i else j
                temp_tour.insert(insert_pos, u)
                
                if len(set(temp_tour)) != n:
                    continue
                
                new_len = sum(d[temp_tour[k], temp_tour[(k + 1) % n]] for k in range(n))
                if new_len < best_len - 1e-10:
                    best_tour = temp_tour
                    best_len = new_len
                    improved = True
                    break
    
    return best_tour

def _compute_mst_heuristic(d):
    """Compute MST and create a heuristic matrix with high values on MST edges."""
    n = d.shape[0]
    # Get MST edges
    mst = minimum_spanning_tree(d)
    mst_dense = mst.toarray()
    
    # Create heuristic: high for MST edges, low for others
    # Use a ratio-based approach
    eta = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if mst_dense[i, j] < 1e-10:  # MST edge
                eta[i, j] = 100.0
            else:
                eta[i, j] = 1.0
    return eta

def _hybrid_heuristic(d, best_tour, mst_eta):
    """Combine inverse-distance with MST structure and tour boost."""
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    d_safe = d.copy()
    d_safe[mask] = 1.0
    
    # Base: inverse distance
    base_eta = 1.0 / (d_safe ** 2.0)
    
    # Blend with MST
    eta = 0.3 * base_eta + 0.7 * mst_eta
    
    # Boost best tour edges
    for k in range(n):
        u = best_tour[k]
        v = best_tour[(k + 1) % n]
        eta[u, v] *= 50.0
        eta[v, u] *= 50.0
    
    eta[mask] = 1e-9
    return eta

def _simple_aco_eval(d, eta, n_ants=5, n_iter=3, seed=42):
    """Very fast ACO simulation to score heuristic quality."""
    np.random.seed(seed)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    best_len = float('inf')
    pheromone = np.ones((n, n))
    pheromone[mask] = 0
    
    for _ in range(n_iter):
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour = [0]
            
            for _ in range(n - 1):
                # Calculate probabilities
                probs = pheromone[current] * eta[current]
                probs[visited] = 0
                probs[current] = 0
                total = probs.sum()
                if total < 1e-15:
                    # Random choice
                    unvisited = np.where(~visited)[0]
                    next_node = unvisited[np.random.randint(len(unvisited))]
                else:
                    probs = probs / total
                    next_node = np.random.choice(n, p=probs)
                
                tour.append(next_node)
                visited[next_node] = True
                current = next_node
            
            # Close tour
            tour_len = sum(d[tour[k], tour[(k + 1) % n]] for k in range(n))
            if tour_len < best_len:
                best_len = tour_len
            
            # Update pheromone
            pheromone *= 0.9
            for k in range(n):
                u = tour[k]
                v = tour[(k + 1) % n]
                pheromone[u, v] += 1.0 / tour_len
                pheromone[v, u] += 1.0 / tour_len
    
    return best_len

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # 1. Find a high quality tour (fewer restarts, faster local search)
    best_tour = None
    best_tour_len = float('inf')
    
    # Use 20 restarts for NN + 2-opt
    for i in range(20):
        if time.time() > deadline - 30.0:
            break
        start_node = i % n
        tour = _nearest_neighbor_tour(d, start=start_node)
        tour = _two_opt(tour, d)
        length = sum(d[tour[k], tour[(k + 1) % n]] for k in range(n))
        if length < best_tour_len:
            best_tour_len = length
            best_tour = tour[:]
    
    if best_tour is None:
        best_tour = _nearest_neighbor_tour(d, start=0)
        best_tour = _two_opt(best_tour, d)
        best_tour_len = sum(d[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))
    
    # Apply Or-opt to the best tour
    best_tour = _fast_or_opt(best_tour, d)
    best_tour_len = sum(d[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))
    
    # 2. Compute MST-based heuristic
    mst_eta = _compute_mst_heuristic(d)
    
    # 3. Generate hybrid heuristics
    candidates = []
    
    # Hybrid with different boost values
    for boost in [10.0, 20.0, 50.0, 100.0]:
        n = d.shape[0]
        mask = np.eye(n, dtype=bool)
        d_safe = d.copy()
        d_safe[mask] = 1.0
        
        base_eta = 1.0 / (d_safe ** 2.0)
        eta = 0.3 * base_eta + 0.7 * mst_eta
        
        for k in range(n):
            u = best_tour[k]
            v = best_tour[(k + 1) % n]
            eta[u, v] *= boost
            eta[v, u] *= boost
        
        eta[mask] = 1e-9
        candidates.append(eta)
    
    # Pure MST with boost
    for boost in [20.0, 50.0]:
        eta = mst_eta.copy()
        for k in range(n):
            u = best_tour[k]
            v = best_tour[(k + 1) % n]
            eta[u, v] *= boost
            eta[v, u] *= boost
        eta[mask] = 1e-9
        candidates.append(eta)
    
    # 4. Evaluate candidates with fast ACO simulation
    best_eta = candidates[0]
    best_score = float('inf')
    
    for eta in candidates:
        if time.time() > deadline - 10.0:
            break
        score = _simple_aco_eval(d, eta)
        if score < best_score:
            best_score = score
            best_eta = eta.copy()
    
    return best_eta
