"""Heuristic via EXPERT-SELECTION: evaluate a small set of strong bin-packing
heuristics using a fast deterministic proxy, and select the best one."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _det_greedy_bins(demands, capacity, M):
    """
    Deterministic greedy proxy:
    1. Sort items by demand descending (optional, but we must respect the 'random first' 
       constraint? No, the proxy is just for ranking heuristics. 
       The actual ACO picks a random first item. 
       To be a fair proxy for the ACO's *best* ant, we should simulate the ACO logic 
       but deterministically? 
       Actually, the ACO's performance depends on the heuristic guiding the pheromone. 
       A good heuristic should lead to fewer bins even with random starts.
       
       Let's run a simplified ACO: 
       - 1 ant, 1 iteration.
       - Start with the largest item (worst case for random start is small, but largest is a good anchor).
       - Actually, let's just run the deterministic version of the ACO step:
         Pick j that maximizes H[current, j] among fitting items.
       - We need to handle the "random first item" aspect. 
       - Let's try starting with the largest item, as that is often the bottleneck.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Start with the largest item to simulate a "good" start, or average over a few starts?
    # Averaging is more robust. Let's start with the top 3 largest items and average bins.
    start_indices = np.argsort(d)[-3:]
    
    total_bins = 0
    for start_idx in start_indices:
        used = np.zeros(n, dtype=bool)
        bins = 0
        used[start_idx] = True
        bin_load = d[start_idx]
        bins += 1
        current = start_idx
        
        while True:
            # Find fitting items
            remaining = np.where(~used)[0]
            if len(remaining) == 0:
                break
            
            fitting = []
            for j in remaining:
                if d[j] <= capacity - bin_load:
                    fitting.append(j)
            
            if len(fitting) == 0:
                break
            
            # Pick j that maximizes M[current, j]
            # If tie, pick largest demand? 
            best_j = -1
            best_score = -1
            for j in fitting:
                score = M[current, j]
                if score > best_score:
                    best_score = score
                    best_j = j
            
            if best_j == -1:
                break
                
            used[best_j] = True
            bin_load += d[best_j]
            current = best_j
            
        # Check if any items remain
        if np.any(~used):
            # Open new bin with the largest remaining item
            remaining = np.where(~used)[0]
            next_start = remaining[np.argmax(d[remaining])]
            used[next_start] = True
            bin_load = d[next_start]
            bins += 1
            current = next_start
            # Continue filling this bin? The loop above breaks when no fitting item.
            # We need to re-enter the filling loop.
            # Refactor: use a while loop for bins.
            
    # The logic above is slightly broken for multiple bins. Let's rewrite properly.
    
    total_bins = 0
    for start_idx in start_indices:
        used = np.zeros(n, dtype=bool)
        bins = 0
        
        while True:
            remaining = np.where(~used)[0]
            if len(remaining) == 0:
                break
            
            # Start new bin with largest remaining item (greedy FFD style start)
            # This is a proxy for "random start" but biased towards difficult items.
            # Actually, the task says "places a random fitting item". 
            # To be robust, we should perhaps just use the largest item as a deterministic proxy for the "best" ant's start?
            # Or just pick the first available? 
            # Let's stick to: Start bin with largest remaining.
            start = remaining[np.argmax(d[remaining])]
            used[start] = True
            bin_load = d[start]
            bins += 1
            current = start
            
            while True:
                # Find fitting items
                rem = np.where(~used)[0]
                fitting = [j for j in rem if d[j] <= capacity - bin_load]
                
                if len(fitting) == 0:
                    break
                
                # Pick best j based on M
                best_j = -1
                best_score = -1
                for j in fitting:
                    score = M[current, j]
                    if score > best_score:
                        best_score = score
                        best_j = j
                
                if best_j == -1:
                    break
                    
                used[best_j] = True
                bin_load += d[best_j]
                current = best_j
                
        total_bins += bins
        
    return total_bins / len(start_indices)


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    capacity = float(capacity)
    n = len(demands)
    
    # Candidate Heuristics
    # 1. Tight Fit: 1 / (C - sum + eps). Favors completing the bin.
    # 2. Gaussian: exp(-alpha * (sum - C)^2). Favors near-completion.
    # 3. Linear Fill: sum. Favors heavy pairs.
    # 4. Inverse Empty: 1 / (empty + eps). Same as Tight Fit essentially.
    # 5. Constant 1. (Baseline)
    
    candidates = {}
    
    # 1. Tight Fit
    M1 = np.full((n, n), 1e-9)
    pair = demands[:, None] + demands[None, :]
    ok = pair <= capacity
    empty = capacity - pair
    # Avoid division by zero
    M1[ok] = 1.0 / (empty[ok] + 1.0)
    candidates['tight'] = M1
    
    # 2. Gaussian (soft=0.1)
    M2 = np.full((n, n), 1e-9)
    M2[ok] = np.exp(-0.1 * (pair[ok] - capacity) ** 2)
    candidates['gauss'] = M2
    
    # 3. Linear Fill
    M3 = np.full((n, n), 1e-9)
    M3[ok] = pair[ok]
    candidates['linear'] = M3
    
    # 4. Best Fit Decreasing Proxy: 
    #    In BFD, you pick the item that leaves the least space.
    #    This is similar to Tight Fit, but let's try a sharper version.
    M4 = np.full((n, n), 1e-9)
    M4[ok] = 1.0 / (empty[ok] + 0.1)
    candidates['tight_sharp'] = M4
    
    # 5. Hybrid: Tight + Linear
    M5 = np.full((n, n), 1e-9)
    M5[ok] = (1.0 / (empty[ok] + 1.0)) + (pair[ok] / capacity)
    candidates['hybrid'] = M5
    
    # Evaluate each candidate
    best_name = 'tight'
    best_score = float('inf')
    
    for name, M in candidates.items():
        # Safety check
        if not np.all(np.isfinite(M)):
            M[~np.isfinite(M)] = 1e-9
        score = _det_greedy_bins(demands, capacity, M)
        if score < best_score:
            best_score = score
            best_name = name
            
    # Return the best matrix
    M_best = candidates[best_name]
    M_best[~np.isfinite(M_best)] = 1e-9
    M_best[M_best <= 0] = 1e-9
    return M_best
# EVOLVE-BLOCK-END
