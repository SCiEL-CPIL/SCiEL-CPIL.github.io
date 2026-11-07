import numpy as np
import time

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Implements directed pheromone updates as per contract.
    """
    rng = np.random.RandomState(seed)
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
                    next_node = unvisited_indices[rng.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
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
    deadline = start_time + 290.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Search for optimal beta over a wider range with more simulation
    # First pass: coarse search
    betas = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    best_beta = 2.0
    best_score = float('inf')
    
    for beta in betas:
        if time.time() > deadline - 30.0:
            break
            
        h = base_heur ** beta
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        
        score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=30, seed=42)
        
        if score < best_score:
            best_score = score
            best_beta = beta
            
    # Fine-grained search around best_beta
    if time.time() < deadline - 40.0:
        lower = max(0.5, best_beta - 1.0)
        upper = min(6.0, best_beta + 1.0)
        fine_betas = np.linspace(lower, upper, 15)
        
        for beta in fine_betas:
            if time.time() > deadline - 30.0:
                break
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            h[~np.isfinite(h)] = 1e-9
            score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=30, seed=42)
            
            if score < best_score:
                best_score = score
                best_beta = beta
                
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
