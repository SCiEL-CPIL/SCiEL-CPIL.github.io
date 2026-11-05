import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # 1. Generate powerful numbers up to L_BASE
    L_BASE = 10**7
    powerful_set = set()
    # b^3 <= L_BASE
    max_b = isqrt(L_BASE) # actually cbrt, but isqrt is fine for upper bound
    # More precise: b <= (10^7)^(1/3) ~ 215
    for b in range(1, 216):
        b3 = b ** 3
        if b3 > L_BASE: break
        # a^2 <= L_BASE / b3
        max_a = isqrt(L_BASE // b3)
        for a in range(1, max_a + 1):
            p = a * a * b3
            if p <= L_BASE:
                powerful_set.add(p)
    
    powerful_list = sorted(powerful_set)
    
    # 2. Count representations for sums up to 2 * L_BASE
    max_sum = 2 * L_BASE
    sum_counts = {}
    sum_pairs = {}
    
    n_pow = len(powerful_list)
    for i in range(n_pow):
        a = powerful_list[i]
        if a > max_sum: break
        for j in range(i, n_pow):
            b = powerful_list[j]
            s = a + b
            if s > max_sum:
                break
            if s not in sum_pairs:
                sum_pairs[s] = []
            sum_pairs[s].append((a, b))
            
    # Check time
    if time.time() > deadline - 10:
         # Fallback to a simple known good N
         # N=2, pair (1,1)
         return {'N': 2, 'pairs': [[1, 1]]}

    # 3. Find best S
    best_s = 2
    max_count = 1
    for s, pairs in sum_pairs.items():
        if len(pairs) > max_count:
            max_count = len(pairs)
            best_s = s
            
    # 4. Scale
    # N = best_s * k^2 <= 10^18
    k = isqrt(10**18 // best_s)
    N = best_s * k * k
    
    # 5. Construct pairs
    pairs = []
    for a, b in sum_pairs[best_s]:
        a_scaled = a * k * k
        b_scaled = b * k * k
        pairs.append([a_scaled, b_scaled])
        
    # Sort pairs to ensure distinct and a<=b (already a<=b from gen)
    # The problem requires distinct pairs.
    # Since a,b are distinct in the loop (i<=j), and scaling is injective, pairs are distinct.
    
    return {'N': N, 'pairs': pairs}
