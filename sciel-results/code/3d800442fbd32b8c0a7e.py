import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute conflict masks
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in numbers:
        for b in numbers:
            if a == b:
                continue
            g_ab = gcd(a, b)
            mask = 0
            for c in numbers:
                if c == a or c == b:
                    continue
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    mask |= (1 << (c - 1))
            conflict_mask[a][b] = mask

    def is_valid_add(S_mask, x):
        for a in numbers:
            if S_mask & (1 << (a - 1)):
                if conflict_mask[a][x] & S_mask:
                    return False
        return True

    def greedy_build(order):
        S_mask = 0
        for x in order:
            if is_valid_add(S_mask, x):
                S_mask |= (1 << (x - 1))
        return S_mask

    def mask_to_list(S_mask):
        return [x for x in numbers if S_mask & (1 << (x - 1))]

    best_mask = 0
    best_size = 0
    
    S0 = greedy_build(numbers)
    best_mask = S0
    best_size = bin(S0).count("1")
    
    def local_search(S_mask):
        current_mask = S_mask
        current_size = bin(current_mask).count("1")
        
        while time.time() < deadline - 2.0:
            improved = False
            S_list = mask_to_list(current_mask)
            unselected = [x for x in numbers if not (current_mask & (1 << (x - 1)))]
            
            if not unselected:
                break
                
            random.shuffle(S_list)
            random.shuffle(unselected)
            
            # Limit attempts to avoid long stalls
            attempts = 0
            max_attempts = 100
            
            # Try 1-for-2 moves
            for u in S_list:
                if attempts > max_attempts or time.time() > deadline - 2.0:
                    break
                attempts += 1
                temp_mask = current_mask & ~(1 << (u - 1))
                
                for v in unselected:
                    if attempts > max_attempts or time.time() > deadline - 2.0:
                        break
                    if not is_valid_add(temp_mask, v):
                        continue
                    attempts += 1
                    temp_mask_v = temp_mask | (1 << (v - 1))
                    
                    for w in unselected:
                        if w == v:
                            continue
                        if attempts > max_attempts or time.time() > deadline - 2.0:
                            break
                        if not is_valid_add(temp_mask_v, w):
                            continue
                        
                        new_mask = temp_mask_v | (1 << (w - 1))
                        current_mask = new_mask
                        current_size += 1
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
                    
            if not improved:
                # Try 1-for-1
                for u in S_list:
                    if time.time() > deadline - 2.0:
                        return current_mask
                    temp_mask = current_mask & ~(1 << (u - 1))
                    for v in unselected:
                        if is_valid_add(temp_mask, v):
                            current_mask = temp_mask | (1 << (v - 1))
                            improved = True
                            break
                    if improved:
                        break
                        
            if not improved:
                break
                
        return current_mask

    for _ in range(1000):
        if time.time() > deadline - 5.0:
            break
            
        random_order = numbers.copy()
        random.shuffle(random_order)
        S_greedy = greedy_build(random_order)
        
        S_local = local_search(S_greedy)
        size = bin(S_local).count("1")
        
        if size > best_size:
            best_size = size
            best_mask = S_local

    return mask_to_list(best_mask)
