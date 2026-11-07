import numpy as np
import time

def _run_aco_sim(dist, eta, n_ants=20, n_iters=30, seed=42):
    # Kept for signature compatibility, but we use _run_aco_sim_optimized
    return _run_aco_sim_optimized(dist, eta, n_ants, n_iters, seed)

def _run_aco_sim_optimized(dist, eta, n_ants=20, n_iters=30, seed=42):
    n = len(dist)
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    
    best_len = float('inf')
    
    for _ in range(n_iters):
        tours_list = []
        
        for _ in range(n_ants):
            tour = np.empty(n + 1, dtype=np.int32)
            tour[0] = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            curr = 0
            
            for step in range(1, n):
                probs = tau[curr, :] * eta[curr, :]
                probs[visited] = 0.0
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    unvisited = np.where(~visited)[0]
                    next_node = rng.choice(unvisited)
                else:
                    probs /= total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour[step] = next_node
                visited[next_node] = True
                curr = next_node
            
            tour[n] = 0
            
            length = np.sum(dist[tour[:-1], tour[1:]])
            
            if length < best_len:
                best_len = length
            
            tours_list.append((length, tour))
        
        tau *= 0.9
        
        for length, tour_arr in tours_list:
            reward = 1.0 / length
            u = tour_arr[:-1]
            v = tour_arr[1:]
            tau[u, v] += reward

    return best_len

def heuristics(distance_matrix):
    start_time = time.time()
    timeout = 290
    
    n = len(distance_matrix)
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    
    alphas = np.geomspace(0.5, 100.0, 40)
    seeds = [1, 2, 3]
    
    best_alpha = 1.0
    best_score = float('inf')
    
    for alpha in alphas:
        if time.time() - start_time > timeout:
            break
            
        eta = 1.0 / (safe_dist ** alpha)
        np.fill_diagonal(eta, 1e-9)
        
        total_score = 0.0
        for s in seeds:
            # Use lighter simulation for search
            score = _run_aco_sim_optimized(distance_matrix, eta, n_ants=10, n_iters=10, seed=s)
            total_score += score
            if time.time() - start_time > timeout:
                break
                
        avg_score = total_score / len(seeds)
        
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = alpha

    eta = 1.0 / (safe_dist ** best_alpha)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
