import numpy as np
import time
import random

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    num_targets = len(targets)
    
    # Precompute pair sums? No, we can use convolution for speed.
    
    def get_metrics(x):
        """
        Returns (missing_count, max_mult)
        x: numpy array of 0s and 1s of length n
        """
        conv = np.convolve(x, x, mode='full')
        conv_vals = conv[targets]
        
        even_mask = (targets % 2 == 0)
        odd_mask = ~even_mask
        
        R = np.zeros(num_targets)
        
        # Odd targets: conv is even, R = conv / 2
        R[odd_mask] = conv_vals[odd_mask] / 2.0
        
        # Even targets: R = (conv + x[mid]) / 2
        mid_indices = targets[even_mask] // 2
        mid_vals = x[mid_indices]
        R[even_mask] = (conv_vals[even_mask] + mid_vals) / 2.0
        
        # R should be integer. Convert to int to be safe against float issues
        R = R.astype(np.int64)
        
        missing = np.sum(R == 0)
        max_mult = np.max(R) if num_targets > 0 else 0
        
        return missing, max_mult

    def get_cost(x):
        missing, max_mult = get_metrics(x)
        # Strong penalty for missing coverage
        return missing * 10000 + max_mult

    # --- Phase 1: Greedy Construction ---
    
    def greedy_construct():
        x = np.zeros(n, dtype=np.int8)
        
        # Start with a few random elements to break symmetry?
        # Or start empty.
        
        while True:
            missing, max_mult = get_metrics(x)
            if missing == 0:
                break
            
            best_candidate = -1
            best_cost = np.inf
            best_missing = np.inf
            best_max_mult = np.inf
            
            # Try adding each element not in A
            # To speed up, we can just iterate through all 0..128
            for i in range(n):
                if x[i] == 1:
                    continue
                
                x[i] = 1
                m, mm = get_metrics(x)
                x[i] = 0
                
                # Priority: minimize missing, then minimize max_mult
                if m < best_missing or (m == best_missing and mm < best_max_mult):
                    best_missing = m
                    best_max_mult = mm
                    best_candidate = i
                    best_cost = m * 10000 + mm
            
            if best_candidate == -1:
                break
            
            x[best_candidate] = 1
            
        return x

    x = greedy_construct()
    
    # --- Phase 2: Local Search (Hill Climbing) ---
    
    current_cost = get_cost(x)
    current_missing, current_max_mult = get_metrics(x)
    
    # We want to minimize max_mult, but keep missing == 0.
    # So we define cost as: if missing > 0, cost = 10000*missing + max_mult
    # else cost = max_mult
    
    # Hill Climbing with First Improvement
    # We will try to reduce max_mult by swapping elements or removing redundant ones.
    
    time_limit = 50.0 # Leave 10s buffer
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_neighbor_cost = current_cost
        best_neighbor_x = None
        
        # Try removals
        for i in range(n):
            if x[i] == 1:
                x[i] = 0
                new_cost = get_cost(x)
                if new_cost < best_neighbor_cost:
                    best_neighbor_cost = new_cost
                    best_neighbor_x = x.copy()
                x[i] = 1
                
        # Try additions (if we have space and it helps? Usually we want smaller sets, but maybe a larger set with lower peak? 
        # Actually, adding elements usually increases peak. Removing is key.
        # But sometimes swapping is better.)
        
        # Try swaps: Remove i, Add j
        # This is O(N^2) which is 129^2 ~ 16k per iteration. 
        # get_cost is O(N log N) or O(N^2) due to conv. 
        # 16k * 129 operations is ~ 2M ops per iteration. 
        # Python is slow. We should be careful.
        
        # Let's optimize get_cost or limit swaps.
        # Actually, let's just do single-bit flips first.
        
        if best_neighbor_x is not None:
            x = best_neighbor_x
            current_cost = best_neighbor_cost
            improved = True
            continue
            
        # If no single-bit improvement, try swaps
        # To save time, we can sample swaps or limit them.
        # Let's try a few random swaps per iteration to avoid O(N^2) full scan if it's too slow.
        # Or just do full scan if it's fast enough.
        
        # Let's check speed of get_cost.
        # np.convolve on 129 elements is very fast (~microseconds).
        # 16k calls * 10us = 0.16s per full swap scan.
        # This is acceptable for 50s limit (approx 300 full scans).
        
        for i in range(n):
            if x[i] == 0:
                continue
            x[i] = 0
            for j in range(n):
                if x[j] == 1:
                    continue
                x[j] = 1
                new_cost = get_cost(x)
                if new_cost < best_neighbor_cost:
                    best_neighbor_cost = new_cost
                    # Save state
                    best_neighbor_x = x.copy()
                x[j] = 0
            x[i] = 1
            
        if best_neighbor_x is not None:
            x = best_neighbor_x
            current_cost = best_neighbor_cost
            improved = True

    # Final check: ensure non-empty and valid
    missing, max_mult = get_metrics(x)
    if missing > 0:
        # Fallback: if greedy failed or local search broke coverage, use a known valid set
        # A simple valid set: A = [0, 128] doesn't cover all.
        # A = [64, 65, ..., 128, 0, 1, ..., 63]?
        # Actually, if missing > 0, we should just return a set that works, even if suboptimal.
        # Let's try to fix it quickly by adding elements.
        while missing > 0 and (time.time() - start_time) < time_limit:
            best_candidate = -1
            best_missing = np.inf
            for i in range(n):
                if x[i] == 1:
                    continue
                x[i] = 1
                m, _ = get_metrics(x)
                x[i] = 0
                if m < best_missing:
                    best_missing = m
                    best_candidate = i
            if best_candidate == -1:
                break
            x[best_candidate] = 1
            missing, _ = get_metrics(x)
            
    result = np.nonzero(x)[0].tolist()
    if not result:
        return [64]
        
    return result
