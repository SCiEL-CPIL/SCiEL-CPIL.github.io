import math
from collections import defaultdict

def construct():
    limit = 10**8
    powerful = set()
    
    max_b = int(round(limit ** (1.0/3.0))) + 1
    
    for b in range(1, max_b + 1):
        b_cubed = b ** 3
        if b_cubed > limit:
            break
        max_a = int(math.isqrt(limit // b_cubed))
        for a in range(1, max_a + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful.add(val)
    
    powerful_list = sorted(powerful)
    p_len = len(powerful_list)
    
    sum_counts = defaultdict(int)
    
    # Count sums
    # To speed up, we can break inner loop if a+b > some threshold?
    # No, we want the max count, which could be anywhere.
    # But we know r(N) is decreasing, so max is likely at small N.
    # However, we must count all to be sure.
    
    for i in range(p_len):
        a = powerful_list[i]
        for j in range(i, p_len):
            b = powerful_list[j]
            s = a + b
            sum_counts[s] += 1
            
    best_N = 0
    max_count = 0
    for N, count in sum_counts.items():
        if count > max_count:
            max_count = count
            best_N = N
            
    pairs = []
    powerful_set = powerful # For O(1) lookup
    
    for i in range(p_len):
        a = powerful_list[i]
        b = best_N - a
        if b < a:
            continue
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
