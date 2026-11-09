import random
import time

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N_MAX = 192
    TARGET_SIZE = 48
    
    def is_valid_addition(S, x):
        S_list = list(S)
        n = len(S_list)
        
        # Case 1: x is divisor
        # Check if any triple in S sums to k*x
        if n >= 3:
            min_sum = 3 * min(S_list)
            max_sum = 3 * max(S_list)
            min_k = max(1, (min_sum + x - 1) // x)
            max_k = max_sum // x
            
            S_set = set(S_list)
            # Sort for potential pruning or just use set
            for k in range(min_k, max_k + 1):
                target = k * x
                # Check if any triple sums to target
                # Iterate i, j
                # To avoid duplicates, i < j < l
                # But we just need existence.
                # Using set lookup for l = target - Si - Sj
                # We must ensure l != Si, l != Sj, l in S
                found = False
                for i in range(n):
                    si = S_list[i]
                    if si > target: break # if sorted
                    for j in range(i + 1, n):
                        sj = S_list[j]
                        if si + sj > target: break
                        sl = target - si - sj
                        if sl <= sj: continue # ensures distinct indices if sorted and sl > sj
                        if sl in S_set:
                            # Ensure distinctness from si, sj
                            # sl > sj > si, so distinct.
                            found = True
                            break
                    if found: break
                if found:
                    return False
        
        # Case 2: x is summand
        # For each a in S, check if x + b + c is multiple of a
        S_set = set(S_list)
        for a in S_list:
            # b, c in S \ {a}
            # x + b + c = k * a
            # b + c = k * a - x
            # Min b+c: 2 * min(S\{a})
            # Max b+c: 2 * max(S\{a})
            # Actually just iterate b
            for b in S_list:
                if b == a: continue
                # c = k*a - x - b
                # We need c in S, c != a, c != b, c != x (x not in S)
                # Range of c: 1..192
                # k*a = x + b + c
                # k = (x + b + c) / a
                # c = k*a - x - b
                # Min c = 1 => k*a >= x + b + 1 => k >= (x+b+1)/a
                # Max c = 192 => k*a <= x + b + 192 => k <= (x+b+192)/a
                min_k = max(1, (x + b + 1 + a - 1) // a)
                max_k = (x + b + 192) // a
                
                for k in range(min_k, max_k + 1):
                    c = k * a - x - b
                    if 1 <= c <= 192:
                        if c in S_set:
                            if c != a and c != b:
                                return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if is_valid_addition(S, x):
                S.add(x)
                if len(S) >= TARGET_SIZE:
                    return S
        return S

    # 1. Deterministic large-first
    order = list(range(N_MAX, 0, -1))
    S = greedy_construct(order)
    if len(S) >= TARGET_SIZE:
        return list(S)
    
    best_S = S
    best_size = len(S)
    
    # 2. Random restarts
    while time.time() - start_time < timeout:
        order = list(range(1, N_MAX + 1))
        random.shuffle(order)
        S_new = greedy_construct(order)
        
        if len(S_new) > best_size:
            best_size = len(S_new)
            best_S = S_new
            if best_size >= TARGET_SIZE:
                return list(best_S)

    return list(best_S)
