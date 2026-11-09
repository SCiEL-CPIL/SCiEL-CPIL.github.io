import numpy as np
import time

def simulate_aco(demands, capacity, H, seed):
    np.random.seed(seed)
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    # Optimization: Precompute feasible items for each item to speed up inner loop?
    # Actually, the inner loop iterates over remaining items.
    # Let's keep the structure but ensure it's fast enough.
    # 20 ants * 30 iters * 120 items * ~60 avg remaining checks = ~4.3M operations.
    # Python loops are slow. We should try to vectorize the selection or accept the speed if it's < 10s per seed.
    # 3 seeds * 10s = 30s. This is safe.
    
    for _ in range(30):
        bins_counts = []
        for _ in range(20):
            remaining = list(range(n))
            bins = 0
            current_bin_items = []
            current_load = 0
            last_item = -1
            
            while remaining:
                if not current_bin_items:
                    # Random fitting item
                    idx = np.random.randint(len(remaining))
                    i = remaining[idx]
                    remaining.pop(idx)
                    current_bin_items.append(i)
                    current_load = demands[i]
                    last_item = i
                    bins += 1
                else:
                    # Find feasible j
                    feasible_indices = []
                    weights = []
                    # Iterate through remaining items
                    for k, j in enumerate(remaining):
                        if current_load + demands[j] <= capacity:
                            feasible_indices.append(k)
                            # prob proportional to tau[last_item, j] * H[last_item, j]
                            w = tau[last_item, j] * H[last_item, j]
                            if w <= 0: 
                                w = 1e-9
                            weights.append(w)
                    
                    if not feasible_indices:
                        # Open new bin
                        current_bin_items = []
                        current_load = 0
                        last_item = -1
                        continue
                    
                    # Select j
                    total_w = sum(weights)
                    r = np.random.rand() * total_w
                    cumulative = 0
                    chosen_k = feasible_indices[0]
                    for k_idx in range(len(feasible_indices)):
                        cumulative += weights[k_idx]
                        if r <= cumulative:
                            chosen_k = feasible_indices[k_idx]
                            break
                    
                    j = remaining[chosen_k]
                    remaining.pop(chosen_k)
                    current_bin_items.append(j)
                    current_load += demands[j]
                    last_item = j
            
            bins_counts.append(bins)
        
        best_this_iter = min(bins_counts)
        if best_this_iter < best_overall:
            best_overall = best_this_iter
        
        # Update Pheromone
        tau *= 0.9
        tau += 1.0 / best_this_iter
        
    return best_overall

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290 # Safety margin for 330s kill, 40s buffer
    
    n = len(demands)
    d = np.array(demands)
    
    # Precompute pair sums and ratios
    pair_sum = d[:, None] + d[None, :]
    feasible_mask = pair_sum <= capacity
    ratio = pair_sum / capacity
    
    candidates = []
    
    # Candidate 0: Baseline (Ratio^1) - Inspiration 2
    h0 = np.where(feasible_mask, ratio, 1e-9)
    candidates.append(h0)
    
    # Candidate 1: Ratio^2
    h1 = np.where(feasible_mask, ratio**2, 1e-9)
    candidates.append(h1)
    
    # Candidate 2: Ratio^4 - Inspiration 1
    h2 = np.where(feasible_mask, ratio**4, 1e-9)
    candidates.append(h2)
    
    # Candidate 3: Ratio^8
    h3 = np.where(feasible_mask, ratio**8, 1e-9)
    candidates.append(h3)
    
    # Candidate 4: Ratio^16
    h4 = np.where(feasible_mask, ratio**16, 1e-9)
    candidates.append(h4)
    
    # Candidate 5: Inverse Waste (1 / (capacity - pair_sum + eps))
    # This favors tight fits but is unbounded if waste is small.
    # Adding eps=1 prevents division by zero and limits max value.
    waste = capacity - pair_sum
    h5 = np.where(feasible_mask, 1.0 / (waste + 1.0), 1e-9)
    candidates.append(h5)
    
    # Candidate 6: Best Fit Style (1 / waste^2)
    h6 = np.where(feasible_mask, 1.0 / (waste**2 + 1.0), 1e-9)
    candidates.append(h6)
    
    # Candidate 7: Exp(-waste)
    h7 = np.where(feasible_mask, np.exp(-waste / 5.0), 1e-9)
    candidates.append(h7)
    
    best_score = float('inf')
    best_h = h0 # Fallback to baseline
    
    # Use a few seeds to evaluate robustness
    seeds = [42, 123, 7]
    
    for idx, h in enumerate(candidates):
        # Check time
        if time.time() > deadline - 30: 
            break
            
        # Evaluate candidate
        scores = []
        for seed in seeds:
            score = simulate_aco(d, capacity, h, seed)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_h = h
            
        # If we hit a very low score (e.g., close to theoretical lower bound), we can stop early
        # Lower bound is ceil(sum(demands) / capacity)
        lower_bound = np.ceil(np.sum(d) / capacity)
        if best_score <= lower_bound + 1:
            break

    return best_h
