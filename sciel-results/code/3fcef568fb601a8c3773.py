import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    n = 256
    
    # 1. Precompute conflict bitsets
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(n):
            if i == j: continue
            b = j + 1
            s = a + b
            p = a * b
            if p % s == 0:
                conflict[i] |= (1 << j)
                
    def count_bits(x):
        return bin(x).count('1')

    def greedy_add(mask):
        for v in range(n):
            if not (mask & (1 << v)):
                if not (mask & conflict[v]):
                    mask |= (1 << v)
        return mask

    def local_search(initial_mask, time_limit=5.0):
        mask = greedy_add(initial_mask)
        current_size = count_bits(mask)
        last_imp = time.time()
        failed_shakes = 0
        
        while (time.time() - start_time) < deadline and (time.time() - last_imp) < time_limit:
            improved = False
            
            # 1. Try 1-swaps
            for v in range(n):
                if mask & (1 << v):
                    continue
                confs = mask & conflict[v]
                if not confs:
                    mask |= (1 << v)
                    current_size += 1
                    improved = True
                    last_imp = time.time()
                    break
                if confs & (confs - 1) == 0:
                    u_bit = confs
                    new_mask = (mask & ~u_bit) | (1 << v)
                    new_mask = greedy_add(new_mask)
                    new_size = count_bits(new_mask)
                    if new_size > current_size:
                        mask = new_mask
                        current_size = new_size
                        improved = True
                        last_imp = time.time()
                        break
            
            if improved:
                continue
                
            # 2. Shake
            if failed_shakes > 100:
                break
                
            k = random.randint(2, 6)
            indices = [i for i in range(n) if mask & (1 << i)]
            if len(indices) < k:
                break
            to_remove = random.sample(indices, k)
            shake_mask = mask
            for i in to_remove:
                shake_mask &= ~(1 << i)
            shake_mask = greedy_add(shake_mask)
            shake_size = count_bits(shake_mask)
            
            if shake_size > current_size:
                mask = shake_mask
                current_size = shake_size
                improved = True
                last_imp = time.time()
            else:
                failed_shakes += 1
                
        return mask, current_size

    best_mask = 0
    best_size = 0
    
    # Strategy 1: Start with Odds
    mask_odds = 0
    for i in range(0, n, 2):
        mask_odds |= (1 << i)
        
    m, s = local_search(mask_odds, time_limit=10.0)
    if s > best_size:
        best_size = s
        best_mask = m
        
    # Strategy 2: Random Restarts
    while time.time() - start_time < deadline - 5.0:
        # Generate a valid initial set via greedy on random permutation
        perm = list(range(n))
        random.shuffle(perm)
        valid_mask = 0
        for v in perm:
            if not (valid_mask & conflict[v]):
                valid_mask |= (1 << v)
        
        m, s = local_search(valid_mask, time_limit=2.0)
        if s > best_size:
            best_size = s
            best_mask = m
            
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    if not result:
        return [1]
        
    return result
