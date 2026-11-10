import time
from math import gcd

def construct():
    start_time = time.time()
    
    # 1. Calculate LCM of 2..48
    L = 1
    for i in range(2, 49):
        L = L * i // gcd(L, i)
        
    # 2. Weights
    # Indices 0..46 correspond to denominators 2..48
    weights = [L // (i + 2) for i in range(47)]
    
    # Split
    mid = 23
    weights_A = weights[:mid]
    weights_B = weights[mid:]
    
    # 3. Generate sums
    def gen_sums(weights_list):
        sums = [0]
        for w in weights_list:
            new_sums = []
            for s in sums:
                ns = s + w
                if ns < L:
                    new_sums.append(ns)
            sums.extend(new_sums)
        return sums

    sums_A = gen_sums(weights_A)
    sums_B = gen_sums(weights_B)
    
    # 4. Sort
    sums_A.sort()
    sums_B.sort()
    
    # 5. Two-pointer search
    best_sum = -1
    best_A = 0
    best_B = 0
    
    i = 0
    j = len(sums_B) - 1
    
    while i < len(sums_A) and j >= 0:
        current_sum = sums_A[i] + sums_B[j]
        if current_sum < L:
            if current_sum > best_sum:
                best_sum = current_sum
                best_A = sums_A[i]
                best_B = sums_B[j]
            i += 1
        else:
            j -= 1
            
    # 6. Find masks
    def find_mask(weights_list, target_sum):
        # Simple iteration to find one mask
        # Since N is small (23/24), we can just iterate
        # But 2^24 is 16M, which is fine.
        # Optimization: Use a set of visited sums? No, just find one.
        # We can use a recursive search or iterative.
        # Iterative with bitmasks is fastest.
        n = len(weights_list)
        # We want a mask m such that sum(weights[k] for k where m has bit k) == target_sum
        # We can just iterate all masks?
        # 16M iterations is ~1s.
        for mask in range(1 << n):
            s = 0
            # Unroll or simple loop
            for k in range(n):
                if mask & (1 << k):
                    s += weights_list[k]
                    if s > target_sum:
                        break
            if s == target_sum:
                return mask
        return 0 # Should not happen
        
    mask_A = find_mask(weights_A, best_A)
    mask_B = find_mask(weights_B, best_B)
    
    # 7. Construct indices
    indices = []
    for k in range(len(weights_A)):
        if mask_A & (1 << k):
            indices.append(k)
            
    for k in range(len(weights_B)):
        if mask_B & (1 << k):
            indices.append(mid + k)
            
    return sorted(indices)
