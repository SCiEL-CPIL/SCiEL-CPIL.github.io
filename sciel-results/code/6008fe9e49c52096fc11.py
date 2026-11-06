import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    N = 256
    
    # Precompute conflict pairs
    conflict_pairs = {i: [] for i in range(1, N+1)}
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            c = 2*b - a
            if c <= N and c > b:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
    
    def can_add(mask, x):
        for (y, z) in conflict_pairs[x]:
            if ((mask >> (y-1)) & 1) and ((mask >> (z-1)) & 1):
                return False
        return True
    
    def greedy_construct(order):
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x-1))
        return mask
    
    def mask_to_set(mask):
        return [i+1 for i in range(N) if (mask >> i) & 1]
    
    def set_score(mask):
        s = 0.0
        for i in range(N):
            if (mask >> i) & 1:
                s += 1.0 / (i+1)
        return s
    
    def hill_climb(mask):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add elements (smallest first for max reciprocal gain)
            best_add = -1
            best_add_gain = 0.0
            for x in range(1, N+1):
                if (mask >> (x-1)) & 1:
                    continue
                if can_add(mask, x):
                    gain = 1.0 / x
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = x
            
            if best_add != -1:
                mask |= (1 << (best_add-1))
                improved = True
                continue
                
            # 2. Try swaps: remove x_out, add x_in where x_in < x_out
            S = mask_to_set(mask)
            S_set = set(S)
            x_outs = sorted(S, reverse=True)
            x_ins = sorted([x for x in range(1, N+1) if x not in S_set])
            
            best_swap_out = -1
            best_swap_in = -1
            best_swap_gain = 0.0
            
            for x_out in x_outs:
                mask_without = mask & ~(1 << (x_out-1))
                for x_in in x_ins:
                    if x_in >= x_out:
                        break
                    if can_add(mask_without, x_in):
                        gain = (1.0 / x_in) - (1.0 / x_out)
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_out = x_out
                            best_swap_in = x_in
            
            if best_swap_out != -1:
                mask = (mask & ~(1 << (best_swap_out-1))) | (1 << (best_swap_in-1))
                improved = True
                continue

        return mask
    
    # Phase 1: Greedy in ascending order
    mask = greedy_construct(range(1, N+1))
    mask = hill_climb(mask)
    
    best_mask = mask
    best_score = set_score(mask)
    
    # Phase 2: Random restarts
    random.seed(42)
    while time.time() < deadline - 5.0:
        # Biased random order: small numbers more likely to come first
        # Key = x * random()
        order = list(range(1, N+1))
        keys = [x * random.random() for x in order]
        order = [x for _, x in sorted(zip(keys, order))]
        
        mask_rand = greedy_construct(order)
        mask_rand = hill_climb(mask_rand)
        score_rand = set_score(mask_rand)
        
        if score_rand > best_score:
            best_score = score_rand
            best_mask = mask_rand

    return mask_to_set(best_mask)
