import numpy as np
import time

def _simulate_aco(demands, capacity, alpha, n_ants=20, n_iters=30, seed=42):
    n = len(demands)
    np.random.seed(seed)
    
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    ratio = np.zeros_like(pair_sum)
    ratio[fits] = pair_sum[fits] / capacity
    H = np.where(fits, ratio ** alpha, 1e-9)
    H = np.maximum(H, 1e-9)
    
    tau = np.ones((n, n))
    best_bins = float('inf')
    
    for it in range(n_iters):
        min_bins_this_iter = float('inf')
        
        for ant in range(n_ants):
            bins_used = 0
            remaining_items = np.arange(n)
            current_item = -1
            bin_load = 0.0
            
            while len(remaining_items) > 0:
                if current_item == -1:
                    bins_used += 1
                    bin_load = 0.0
                    idx = np.random.randint(len(remaining_items))
                    chosen = remaining_items[idx]
                    remaining_items = np.delete(remaining_items, idx)
                    current_item = chosen
                    bin_load += demands[chosen]
                else:
                    rem_cap = capacity - bin_load
                    fitting_mask = demands[remaining_items] <= rem_cap
                    fitting_indices = np.where(fitting_mask)[0]
                    
                    if len(fitting_indices) == 0:
                        current_item = -1
                        continue
                    
                    js = remaining_items[fitting_indices]
                    probs = tau[current_item, js] * H[current_item, js]
                    probs = np.maximum(probs, 1e-9)
                    probs = probs / probs.sum()
                    
                    idx_in_fitting = np.random.choice(len(js), p=probs)
                    chosen = js[idx_in_fitting]
                    
                    rem_idx = np.where(remaining_items == chosen)[0][0]
                    remaining_items = np.delete(remaining_items, rem_idx)
                    
                    current_item = chosen
                    bin_load += demands[chosen]
            
            if bins_used < min_bins_this_iter:
                min_bins_this_iter = bins_used
        
        if min_bins_this_iter < best_bins:
            best_bins = min_bins_this_iter
            
        tau = 0.9 * tau + (1.0 / min_bins_this_iter)
        
    return best_bins

def heuristics(demands, capacity):
    start_time = time.time()
    n = len(demands)
    
    # Search for optimal alpha
    # Use fewer alphas if time is tight, but 20 should be fast
    alphas = np.linspace(1.0, 15.0, 20)
    
    best_alpha = 1.0
    best_score = float('inf')
    
    for alpha in alphas:
        if time.time() - start_time > 250:
            break
        score = _simulate_aco(demands, capacity, alpha)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    ratio = np.zeros_like(pair_sum)
    ratio[fits] = pair_sum[fits] / capacity
    H = np.where(fits, ratio ** best_alpha, 1e-9)
    H = np.maximum(H, 1e-9)
    
    return H
