import numpy as np
import time
from scipy.optimize import linear_sum_assignment

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
            remove_cost = d[prev, u] + d[u, next_] - d[prev, next_]
            
            # Try inserting u at all other positions
            for j in range(n - 1):
                if j == (i - 1) % n or j == i:
                    continue
                a = best_tour[j]
                b = best_tour[(j + 1) % n]
                insert_cost = d[a, u] + d[u, b] - d[a, b]
                
                if insert_cost < remove_cost - 1e-10:
                    # Perform the move
                    new_tour = best_tour[:]
                    new_tour.pop(i)
                    # Find position of j in new_tour
                    if j >= i:
                        new_j = j - 1
                    else:
                        new_j = j
                    new_tour.insert(new_j + 1, u)
                    
                    new_len = best_len - remove_cost + insert_cost
                    if new_len < best_len - 1e-10:
                        best_tour = new_tour
                        best_len = new_len
                        improved = True
                        break
        # Try moving 2 nodes (segment of 2)
        for i in range(n):
            u1 = best_tour[i]
            u2 = best_tour[(i + 1) % n]
            prev = best_tour[(i - 1) % n]
            next_ = best_tour[(i + 2) % n]
            
            # Remove segment: cost change
            remove_cost = d[prev, u1] + d[u1, u2] + d[u2, next_] - d[prev, next_]
            
            # Try inserting segment at all other positions
            for j in range(n - 2):
                if j == (i - 1) % n or j == i or j == (i + 1) % n:
                    continue
                a = best_tour[j]
                b = best_tour[(j + 1) % n]
                c = best_tour[(j + 2) % n]
                insert_cost = d[a, u1] + d[u1, u2] + d[u2, b] - d[a, b]
                
                if insert_cost < remove_cost - 1e-10:
                    # Perform the move
                    new_tour = best_tour[:]
                    # Remove u1, u2
                    if i + 1 == n:
                        new_tour.pop(n-1)
                        new_tour.pop(0)
                    else:
                        new_tour.pop(i+1)
                        new_tour.pop(i)
                    
                    # Find position of j in new_tour
                    if j >= i:
                        new_j = j - 2
                    elif j < i:
                        new_j = j
                    
                    new_tour.insert(new_j + 1, u1)
                    new_tour.insert(new_j + 2, u2)
                    
                    new_len = best_len - remove_cost + insert_cost
                    if new_len < best_len - 1e-10:
                        best_tour = new_tour
                        best_len = new_len
                        improved = True
                        break
    return best_tour

def _quick_aco_eval(d, eta, n_ants=20, n_iter=30, seed=42):
    """Quick ACO simulation to evaluate heuristic quality."""
    n = d.shape[0]
    rng = np.random.RandomState(seed)
    
    # Initialize pheromone
    tau = np.ones((n, n))
    
    best_tour_len = float('inf')
    
    for it in range(n_iter):
        tours = []
        for ant in range(n_ants):
            tour = [0]
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            
            for step in range(n - 1):
                # Calculate probabilities
                probs = tau[current] * eta[current]
                probs[visited] = 0
                probs[current] = 0
                
                total = probs.sum()
                if total < 1e-15:
                    # Random choice
                    unvisited = np.where(~visited)[0]
                    next_node = rng.choice(unvisited)
                else:
                    probs = probs / total
                    next_node = rng.choice(n, p=probs)
                
                tour.append(next_node)
                visited[next_node] = True
                current = next_node
            
            # Calculate tour length
            tour_len = sum(d[tour[i], tour[(i + 1) % n]] for i in range(n))
            tours.append((tour, tour_len))
            
            if tour_len < best_tour_len:
                best_tour_len = tour_len
        
        # Update pheromone
        tau *= 0.9
        for tour, tour_len in tours:
            dep = 1.0 / tour_len
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                tau[u, v] += dep
    
    return best_tour_len

def _compute_mwpm_edges(d):
    """Compute maximum-weight perfect matching edges using negative squared distances."""
    n = d.shape[0]
    # For MWPM, we want to maximize sum of weights
    # Use negative squared distance as weight (more negative = shorter distance = better)
    weight_matrix = -d ** 2
    
    # linear_sum_assignment minimizes, so we negate to maximize
    row_ind, col_ind = linear_sum_assignment(-weight_matrix)
    
    # Create edge set
    edges = set()
    for r, c in zip(row_ind, col_ind):
        edges.add((min(r, c), max(r, c)))
    
    return edges

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 300.0
    
    # 1. Compute MWPM edges for structural constraint
    mwpm_edges = _compute_mwpm_edges(d)
    
    # 2. Find a high quality tour
    best_tour = None
    best_tour_len = float('inf')
    
    # Use multiple restarts for NN
    for i in range(30):  # Reduced restarts to save time for MWPM
        if time.time() > deadline - 60.0: break
        start_node = i % n
        tour = _nearest_neighbor_tour(d, start=start_node)
        tour = _two_opt(tour, d)
        tour = _fast_or_opt(tour, d)
        length = sum(d[tour[k], tour[(k + 1) % n]] for k in range(n))
        if length < best_tour_len:
            best_tour_len = length
            best_tour = tour[:]
            
    if best_tour is None:
        best_tour = _nearest_neighbor_tour(d, start=0)
        best_tour = _two_opt(best_tour, d)
        best_tour = _fast_or_opt(best_tour, d)
        best_tour_len = sum(d[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))

    # 3. Generate candidate heuristics with MWPM constraint
    d_safe = d.copy()
    d_safe[mask] = 1.0
    
    candidates = []
    
    # Parameter search
    p_values = [2.0, 2.5, 3.0]
    boost_values = [5.0, 10.0, 20.0]
    mwpm_boost_values = [2.0, 5.0, 10.0]
    
    for p in p_values:
        base_eta = 1.0 / (d_safe ** p)
        for boost in boost_values:
            for mwpm_boost in mwpm_boost_values:
                eta = base_eta.copy()
                
                # Boost tour edges
                if boost > 1.0:
                    for k in range(n):
                        u = best_tour[k]
                        v = best_tour[(k + 1) % n]
                        eta[u, v] *= boost
                        eta[v, u] *= boost
                
                # Boost MWPM edges (structural constraint)
                if mwpm_boost > 1.0:
                    for u, v in mwpm_edges:
                        eta[u, v] *= mwpm_boost
                        eta[v, u] *= mwpm_boost
                
                eta[mask] = 1e-9
                candidates.append(eta)
            
    # 4. Evaluate candidates with quick ACO
    best_eta = candidates[0]
    best_score = float('inf')
    
    for idx, eta in enumerate(candidates):
        if time.time() > deadline - 30.0: break
        score = _quick_aco_eval(d, eta, n_ants=10, n_iter=15, seed=42)
        if score < best_score:
            best_score = score
            best_eta = eta.copy()
            
    return best_eta
