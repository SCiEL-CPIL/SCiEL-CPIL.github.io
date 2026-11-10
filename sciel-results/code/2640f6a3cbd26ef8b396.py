import numpy as np
import time

def _simulate_aco(prizes, distance_matrix, max_len, heuristic, seed):
    np.random.seed(seed)
    n = len(prizes)
    pheromone = np.ones((n, n))
    dist_to_0 = distance_matrix[:, 0]
    
    global_best = 0.0
    
    for _iter in range(30):
        best_prize_this_iter = 0.0
        ant_prizes = [0.0] * 20
        
        for ant_idx in range(20):
            current = 0
            length_so_far = 0.0
            prize_so_far = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                costs = distance_matrix[current, :] + dist_to_0
                feasible_mask = (~visited) & (length_so_far + costs <= max_len)
                
                if not np.any(feasible_mask):
                    break
                
                feasible_indices = np.where(feasible_mask)[0]
                tau_vals = pheromone[current, feasible_indices]
                eta_vals = heuristic[current, feasible_indices]
                
                probs = tau_vals * eta_vals
                prob_sum = np.sum(probs)
                if prob_sum == 0:
                    break
                probs = probs / prob_sum
                
                next_node = np.random.choice(feasible_indices, p=probs)
                
                length_so_far += distance_matrix[current, next_node]
                prize_so_far += prizes[next_node]
                visited[next_node] = True
                current = next_node
            
            ant_prizes[ant_idx] = prize_so_far
            
        best_prize_this_iter = max(ant_prizes)
        if best_prize_this_iter > global_best:
            global_best = best_prize_this_iter
            
        pheromone *= 0.9
        pheromone += best_prize_this_iter / 10.0
        
    return global_best

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
    betas = [1.0, 1.5, 2.0]
    seeds = [1, 2, 3, 4, 5]
    
    best_score = -1
    best_alpha = 1.0
    best_gamma = 1.0
    best_beta = 1.0
    
    # Precompute bases
    p_base = prizes[None, :] + 1e-9
    d_ij_base = distance_matrix + 1e-9
    d_j0_base = distance_matrix[:, 0][None, :] + 1e-9
    
    time_limit_per_instance = 90.0
    
    for alpha in alphas:
        for gamma in gammas:
            for beta in betas:
                if time.time() - start_time > time_limit_per_instance:
                    break
                
                try:
                    eta = (p_base ** beta) / ((d_ij_base ** alpha) * (d_j0_base ** gamma))
                    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                    
                    scores = []
                    for seed in seeds:
                        score = _simulate_aco(prizes, distance_matrix, max_len, eta, seed)
                        scores.append(score)
                    avg_score = np.mean(scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_alpha = alpha
                        best_gamma = gamma
                        best_beta = beta
                        
                except Exception:
                    continue

    # Generate final heuristic
    eta = (p_base ** best_beta) / ((d_ij_base ** best_alpha) * (d_j0_base ** gamma))
    # Fix: use best_gamma
    eta = (p_base ** best_beta) / ((d_ij_base ** best_alpha) * (d_j0_base ** best_gamma))
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
