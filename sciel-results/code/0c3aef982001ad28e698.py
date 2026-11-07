import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin for 90s limit
    
    # Bitmask representation for speed
    # Bit i corresponds to number i+1 (1..128)
    
    def mask_to_set(mask):
        return [i + 1 for i in range(128) if (mask >> i) & 1]
    
    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask

    def can_add(mask, x):
        """Check if adding x to set represented by mask is valid."""
        idx = x - 1
        if (mask >> idx) & 1:
            return False
        
        # Find b in S such that x+b is in S
        max_b = 128 - x
        if max_b < 1:
            return True
            
        A = []
        for b in range(1, max_b + 1):
            b_idx = b - 1
            if (mask >> b_idx) & 1:
                sum_idx = (x + b) - 1
                if (mask >> sum_idx) & 1:
                    A.append(b)
                    
        if len(A) < 2:
            return True
            
        n = len(A)
        for i in range(n):
            b = A[i]
            for j in range(i + 1, n):
                c = A[j]
                s = b + c
                if s == x:
                    return False
                s_idx = s - 1
                if (mask >> s_idx) & 1:
                    return False
        return True

    def is_valid(mask):
        """Full validity check for debugging/fallback."""
        S = mask_to_set(mask)
        S_set = set(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                if (a + b) not in S_set:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S[k]
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True

    best_mask = 0
    best_size = 0
    
    random.seed(42)
    
    def optimize_initial(mask):
        """Greedy fill and local search starting from mask."""
        S_size = bin(mask).count('1')
        
        # 1. Greedy Fill
        changed = True
        while changed:
            changed = False
            for x in range(1, 129):
                if not ((mask >> (x - 1)) & 1):
                    if can_add(mask, x):
                        mask |= (1 << (x - 1))
                        S_size += 1
                        changed = True
        
        # 2. Local Search: 2-for-1 swaps
        improved = True
        while improved:
            improved = False
            S_list = mask_to_set(mask)
            not_in_S = [x for x in range(1, 129) if not ((mask >> (x - 1)) & 1)]
            
            for y in S_list:
                if time.time() > deadline:
                    return mask, S_size
                
                y_idx = y - 1
                mask_temp = mask & ~(1 << y_idx)
                size_temp = S_size - 1
                
                for i in range(len(not_in_S)):
                    x1 = not_in_S[i]
                    if not can_add(mask_temp, x1):
                        continue
                    
                    mask_temp2 = mask_temp | (1 << (x1 - 1))
                    size_temp2 = size_temp + 1
                    
                    for j in range(i + 1, len(not_in_S)):
                        x2 = not_in_S[j]
                        if not can_add(mask_temp2, x2):
                            continue
                        
                        mask = mask_temp2 | (1 << (x2 - 1))
                        S_size = size_temp2 + 1
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
                    
        return mask, S_size

    # Strategies for initialization
    def init_large():
        m = 0
        for x in range(65, 129):
            m |= (1 << (x - 1))
        return m

    def init_odd():
        m = 0
        for x in range(1, 129, 2):
            m |= (1 << (x - 1))
        return m

    def init_random(p):
        m = 0
        for x in range(1, 129):
            if random.random() < p:
                m |= (1 << (x - 1))
        return m

    def init_biased_high():
        m = 0
        for x in range(1, 129):
            p = 0.2 + 0.6 * (x / 128.0)
            if random.random() < p:
                m |= (1 << (x - 1))
        return m

    inits = [
        init_large(),
        init_odd(),
        init_random(0.5),
        init_random(0.3),
        init_biased_high(),
    ]
    
    for m in inits:
        if time.time() > deadline:
            break
        res_mask, res_size = optimize_initial(m)
        if res_size > best_size:
            best_size = res_size
            best_mask = res_mask
    
    # More random restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        m = init_random(random.uniform(0.2, 0.8))
        res_mask, res_size = optimize_initial(m)
        if res_size > best_size:
            best_size = res_size
            best_mask = res_mask

    # Final verification
    if not is_valid(best_mask):
        best_mask = init_large()
        best_size = 64

    result = mask_to_set(best_mask)
    return [int(x) for x in result]
