import numpy as np
import time

def simulate_aco(demands, capacity, H, seed, ants=20, iterations=30):
    np.random.seed(seed)
    n = len(demands)
    d = np.array(demands)
    
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    for _ in range(iterations):
        bins_counts = []
        for _ in range(ants):
            remaining = set(range(n))
            bins = 0
            current_load = 0
            last_item = -1
            
            while remaining:
                if last_item == -1:
                    # Random fitting item
                    items_list = list(remaining)
                    i = items_list[np.random.randint(len(items_list))]
                    remaining.remove(i)
                    current_load = d[i]
                    last_item = i
                    bins += 1
                else:
                    max_allowed = capacity - current_load
                    # Filter remaining
                    feasible_items = [j for j in remaining if d[j] <= max_allowed]
                    
                    if not feasible_items:
                        current_load = 0
                        last_item = -1
                        continue
                    
                    idx_arr = np.array(feasible_items)
                    h_vals = H[last_item, idx_arr]
                    t_vals = tau[last_item, idx_arr]
                    
                    h_vals = np.where(h_vals > 0, h_vals, 1e-9)
                    t_vals = np.where(t_vals > 0, t_vals, 1e-9)
                    
                    weights = t_vals * h_vals
                    weights = np.where(weights > 0, weights, 1e-9)
                    
                    total_w = np.sum(weights)
                    if total_w == 0:
                        chosen_idx = np.random.randint(len(feasible_items))
                    else:
                        probs = weights / total_w
                        chosen_idx = np.random.choice(len(feasible_items), p=probs)
                    
                    j = feasible_items[chosen_idx]
                    remaining.remove(j)
                    current_load += d[j]
                    last_item = j
            
            bins_counts.append(bins)
        
        best_this_iter = min(bins_counts)
        if best_this_iter < best_overall:
            best_overall = best_this_iter
        
        tau *= 0.9
        tau += 1.0 / best_this_iter
        
    return best_overall

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290
    
    n = len(demands)
    d = np.array(demands)
    
    pair_sum = d[:, None] + d[None, :]
    waste = capacity - pair_sum
    feasible_mask = pair_sum <= capacity
    
    d_j_mat = np.tile(d, (n, 1))
    
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0.5, 1.0, 1.5, 2.0]
    gammas = [0.0, 0.5, 1.0]
    
    candidates = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                h = np.zeros((n, n))
                h_vals = (pair_sum ** a) * ((waste + 1.0) ** (-b)) * (d_j_mat ** g)
                h[feasible_mask] = h_vals[feasible_mask]
                h[~feasible_mask] = 1e-9
                candidates.append(h)
                
    test_h = candidates[0]
    t0 = time.time()
    simulate_aco(d, capacity, test_h, 42, ants=10, iterations=10)
    t_sim_small = time.time() - t0
    est_time_full = t_sim_small * 6
    
    max_sims = int(280 / est_time_full) if est_time_full > 0 else 10
    max_sims = max(1, min(max_sims, 50))
    
    seeds = [42, 123]
    num_seeds = len(seeds)
    
    best_score = float('inf')
    best_h = candidates[0]
    
    for idx, h in enumerate(candidates):
        if time.time() > deadline - est_time_full * num_seeds:
            break
            
        scores = []
        for seed in seeds:
            score = simulate_aco(d, capacity, h, seed)
            scores.append(score)
            
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
    return best_h
