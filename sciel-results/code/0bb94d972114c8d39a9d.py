import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    # Base set: [171, 256], size 86
    # Valid because for any a >= 171 in S, b+c <= 512 < 3*171 = 513
    base_S = list(range(171, n + 1))
    best_S = set(base_S)
    best_size = len(base_S)
    
    candidates = list(range(170, 0, -1))  # Decreasing order
    
    # Precompute sum bitmasks: sum_masks[x] has bit k set if x divides k
    # Sums range from 2*1=2 to 2*256=512
    sum_masks = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for k in range(2, 513):
            if k % x == 0:
                mask |= (1 << k)
        sum_masks[x] = mask
    
    # Compute initial sums bitmask for base_S
    # Sums b+c where b <= c in base_S
    base_sums_bits = 0
    L = len(base_S)
    for i in range(L):
        b = base_S[i]
        for j in range(i, L):
            c = base_S[j]
            base_sums_bits |= (1 << (b + c))
    
    # Greedy: add candidates in decreasing order
    # When adding x (which is < all current elements), x is the new minimum
    # So we only need to check: x does not divide any sum of two existing elements
    current_S = base_S[:]
    current_sums_bits = base_sums_bits
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        # Check if x divides any sum in current_sums_bits
        if current_sums_bits & sum_masks[x]:
            continue
        
        # x is safe to add (x is new minimum, so only need case 1)
        current_S.append(x)
        
        # Update sums: add x+y for all y in old current_S
        for y in current_S[:-1]:
            current_sums_bits |= (1 << (x + y))
        
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = set(current_S)
    
    return list(best_S)
