import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix for 3 fixed instances.
    Uses a coarse-to-fine grid search over the exponent gamma in (1/d)^gamma,
    evaluated by a fast proxy ACO.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    def run_proxy_aco(gamma, num_tours):
        """
        Fast proxy ACO using vectorized weight selection.
        Returns the best cost found.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(D > 0, D ** (-gamma), 1e-9)
        
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        best_cost = float('inf')
        rng = np.random.RandomState(int(gamma * 1000))
        
        for t in range(num_tours):
            if time.time() > deadline - 10.0:
                break
            
            current = 0
            current_load = 0.0
            total_distance = 0.0
            unvisited_indices = list(range(1, n))
            
            while unvisited_indices:
                rem_cap = capacity - current_load
                feasible_indices = [j for j in unvisited_indices if demands[j] <= rem_cap]
                
                if not feasible_indices:
                    total_distance += D[current, 0]
                    break
                
                if len(feasible_indices) == 1:
                    next_node = feasible_indices[0]
                else:
                    feasible_arr = np.array(feasible_indices)
                    weights = eta[current, feasible_arr]
                    total_w = np.sum(weights)
                    if total_w <= 0:
                        next_node = feasible_arr[rng.randint(len(feasible_arr))]
                    else:
                        probs = weights / total_w
                        next_node = feasible_arr[rng.choice(len(feasible_arr), p=probs)]
                
                total_distance += D[current, next_node]
                current_load += demands[next_node]
                unvisited_indices.remove(next_node)
                current = next_node
                
                if not unvisited_indices:
                    total_distance += D[current, 0]
                    break
            
            if total_distance < best_cost:
                best_cost = total_distance
        
        return best_cost

    best_gamma = 1.0
    best_proxy_cost = float('inf')
    
    # Phase 1: Coarse search
    coarse_gammas = np.arange(0.5, 3.5, 0.5)
    num_tours_coarse = 200
    
    for gamma in coarse_gammas:
        if time.time() > deadline - 30.0:
            break
        cost = run_proxy_aco(gamma, num_tours_coarse)
        if cost < best_proxy_cost:
            best_proxy_cost = cost
            best_gamma = gamma

    # Phase 2: Fine search around best_gamma
    if time.time() < deadline - 15.0:
        fine_gammas = np.arange(best_gamma - 0.4, best_gamma + 0.4, 0.05)
        fine_gammas = fine_gammas[(fine_gammas > 0.1) & (fine_gammas < 5.0)]
        num_tours_fine = 500
        
        for gamma in fine_gammas:
            if time.time() > deadline - 10.0:
                break
            cost = run_proxy_aco(gamma, num_tours_fine)
            if cost < best_proxy_cost:
                best_proxy_cost = cost
                best_gamma = gamma

    # Construct final heuristic matrix with best_gamma
    with np.errstate(divide='ignore', invalid='ignore'):
        final_heuristic = np.where(D > 0, D ** (-best_gamma), 1e-9)
    
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0), final_heuristic, 1e-9)
    
    if final_heuristic.shape != D.shape:
        final_heuristic = np.ones_like(D) * 1e-9
        np.fill_diagonal(final_heuristic, 1e-9)
        
    return final_heuristic
