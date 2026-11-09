import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin

    N = 512
    # 1. Setup Weights and Sorting
    items = []
    for a in range(1, N):
        w = 1.0 / (N - a)
        items.append((w, a))
    
    # Sort by weight descending
    items.sort(key=lambda x: -x[0])
    
    # Map original value to new index
    new_index = [0] * N
    old_to_new = [0] * N
    weights = [0.0] * N
    for i, (w, a) in enumerate(items):
        new_index[i] = a
        old_to_new[a] = i
        weights[i] = w
        
    # 2. Prime Factors and Conflicts
    def prime_factors(n):
        fs = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                fs.add(d)
                n //= d
            d += 1
        if n > 1:
            fs.add(n)
        return fs
        
    factors = [set() for _ in range(N)]
    for a in range(1, N):
        factors[a] = prime_factors(a)
        
    # Conflict masks in new indexing
    conflict = [0] * N
    for i in range(N):
        ai = new_index[i]
        mask = 0
        for j in range(N):
            if i == j: continue
            aj = new_index[j]
            if factors[ai] & factors[aj]:
                mask |= (1 << j)
        conflict[i] = mask
        
    # 3. Lookup Table for Mask Sum
    NUM_CHUNKS = (N + 15) // 16
    sum_lut = []
    for c in range(NUM_CHUNKS):
        lut = [0.0] * 65536
        for val in range(65536):
            s = 0.0
            for b in range(16):
                if (val >> b) & 1:
                    idx = c * 16 + b
                    if idx < N:
                        s += weights[idx]
            lut[val] = s
        sum_lut.append(lut)
        
    def mask_sum(mask):
        s = 0.0
        for c in range(NUM_CHUNKS):
            chunk = (mask >> (c * 16)) & 0xFFFF
            s += sum_lut[c][chunk]
        return s

    # 4. Greedy Initial Solution
    def greedy():
        mask = 0
        score = 0.0
        for i in range(N):
            if (conflict[i] & mask) == 0:
                mask |= (1 << i)
                score += weights[i]
        return mask, score
        
    best_mask, best_score = greedy()
    
    # 5. Branch and Bound
    initial_avail_mask = (1 << N) - 1
    initial_avail_sum = mask_sum(initial_avail_mask)
    
    stack = [(0, 0, 0.0, initial_avail_mask, initial_avail_sum)]
    
    # Epsilon for float comparison
    EPS = 1e-12
    
    while stack:
        if time.time() > DEADLINE:
            break
            
        k, curr_mask, curr_score, avail_mask, avail_sum = stack.pop()
        
        # Prune
        if curr_score + avail_sum + EPS <= best_score:
            continue
            
        if k >= N:
            if curr_score > best_score:
                best_score = curr_score
                best_mask = curr_mask
            continue
            
        # Branch 2: Exclude k
        # Push to stack first so that Include is processed next (LIFO)
        if (avail_mask >> k) & 1:
            new_avail_excl = avail_mask & ~(1 << k)
            new_sum_excl = avail_sum - weights[k]
        else:
            new_avail_excl = avail_mask
            new_sum_excl = avail_sum
            
        stack.append((k + 1, curr_mask, curr_score, new_avail_excl, new_sum_excl))
        
        # Branch 1: Include k
        if (avail_mask >> k) & 1:
            new_mask_incl = curr_mask | (1 << k)
            new_score_incl = curr_score + weights[k]
            
            new_avail_incl = (avail_mask & ~conflict[k]) & ~(1 << k)
            
            neighbor_sum = mask_sum(conflict[k] & avail_mask)
            new_sum_incl = avail_sum - weights[k] - neighbor_sum
            
            stack.append((k + 1, new_mask_incl, new_score_incl, new_avail_incl, new_sum_incl))

    # Extract result
    result = []
    m = best_mask
    for i in range(N):
        if (m >> i) & 1:
            result.append(new_index[i])
            
    return result
