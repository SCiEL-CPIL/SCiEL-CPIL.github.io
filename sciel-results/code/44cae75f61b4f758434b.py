import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=5, n_iterations=10, rho=0.9):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Correctly implements directed pheromone updates as per contract.
    """
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
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
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Search for optimal beta with wider range
    betas = np.linspace(0.5, 5.0, 20)
    best_beta = 2.0
    best_score = float('inf')
    
    for beta in betas:
        if time.time() > deadline - 15.0:
            break
            
        h = base_heur ** beta
        h[h <= 0] = 1e-9
        
        score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
        
        if score < best_score:
            best_score = score
            best_beta = beta
            
    # Fine-grained search around best_beta
    if time.time() < deadline - 25.0:
        lower = max(0.5, best_beta - 0.5)
        upper = min(5.0, best_beta + 0.5)
        fine_betas = np.linspace(lower, upper, 12)
        
        for beta in fine_betas:
            if time.time() > deadline - 15.0:
                break
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                
    # Try adding local neighborhood enhancement
    if time.time() < deadline - 30.0:
        # Create a k-nearest-neighbor based enhancement
        k = 5
        dist_flat = safe_dist.copy()
        np.fill_diagonal(dist_flat, np.inf)
        
        knn_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            k_indices = np.argsort(dist_flat[i])[:k]
            knn_mask[i, k_indices] = True
        
        # Blend: boost edges to nearest neighbors
        knn_boost = np.where(knn_mask, 2.0, 1.0)
        
        # Test a few blending factors
        for blend in [0.3, 0.5, 0.7]:
            if time.time() > deadline - 15.0:
                break
            
            h = (base_heur ** best_beta) * knn_boost
            h = h / h.max() * base_heur.max()
            h[h <= 0] = 1e-9
            
            score = _fast_aco_simulate(distance_matrix, h, n_ants=5, n_iterations=10)
            
            if score < best_score:
                best_score = score
                # Use this enhanced version
                final_heuristic = h.copy()
                final_heuristic[final_heuristic <= 0] = 1e-9
                final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
                return final_heuristic
                
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
