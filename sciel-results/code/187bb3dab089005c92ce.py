import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9):
    """
    Fast internal ACO simulation matching the contract's parameters (20 ants, 30 iters).
    This is used to evaluate heuristic quality.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = np.zeros(n_ants)
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths[ant] = tour_len
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= rho
        
        current_best = np.min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def _two_opt_improve(tour, dist):
    """
    Apply 2-opt improvement to a tour.
    """
    n = len(tour)
    if n < 4:
        return tour, sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                
                u1 = tour[(i - 1) % n]
                v1 = tour[i]
                u2 = tour[j]
                v2 = tour[(j + 1) % n]
                
                delta = dist[u1, u2] + dist[v1, v2] - dist[u1, v1] - dist[u2, v2]
                if delta < -1e-10:
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    
    total_len = 0.0
    for i in range(n):
        total_len += dist[tour[i], tour[(i + 1) % n]]
    return tour, total_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    # 1. Generate a high-quality tour
    # Nearest Neighbor
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    tour = [0]
    
    for _ in range(n - 1):
        dists = safe_dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    # 2-opt improvement
    tour, _ = _two_opt_improve(tour, safe_dist)
    
    # Create a mask for the tour edges
    tour_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_mask[u, v] = True
        tour_mask[v, u] = True # Symmetric TSP

    # Base heuristic: 1 / d^beta
    # We will search over beta and alpha (boost for tour edges)
    
    best_heur = None
    best_score = float('inf')
    
    # Grid search parameters
    alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    betas = [1.0, 1.5, 2.0, 3.0]
    
    # To save time, use a reduced simulation for the grid search
    # 5 ants, 10 iterations is ~1/12th of the cost
    # 24 candidates * (1/12 cost) = 2 full sim costs. Very safe.
    
    for alpha in alphas:
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            # Construct heuristic
            # H_ij = (1 / d_ij^beta) * (1 + alpha * I(tour))
            # Note: safe_dist has 1.0 on diagonal, so 1/d is 1.0.
            # We want to avoid division by zero or small numbers.
            # safe_dist is already sanitized.
            
            base_h = 1.0 / (safe_dist ** beta)
            
            # Apply boost
            if alpha > 0:
                base_h[tour_mask] *= (1.0 + alpha)
            
            # Sanitize
            base_h[~np.isfinite(base_h)] = 1e-9
            base_h[base_h <= 0] = 1e-9
            
            # Evaluate with reduced simulation
            score = _fast_aco_simulate(safe_dist, base_h, n_ants=5, n_iterations=10)
            
            if score < best_score:
                best_score = score
                best_heur = base_h.copy()
    
    # If no candidate was evaluated (should not happen), fallback to inverse distance
    if best_heur is None:
        best_heur = 1.0 / safe_dist
        best_heur[~np.isfinite(best_heur)] = 1e-9
        best_heur[best_heur <= 0] = 1e-9
        
    return best_heur
