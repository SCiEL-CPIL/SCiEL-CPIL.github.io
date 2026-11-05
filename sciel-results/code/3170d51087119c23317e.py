import time
from math import isqrt
from collections import Counter

def construct():
    start_time = time.time()
    timeout = 110 # Safety margin
    
    # Target M
    # Previous best at 80M was 5.93.
    # Let's try 120M.
    M = 120_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            if p <= M:
                powerful.add(p)
            
    powerful_list = sorted(powerful)
    n_list = len(powerful_list)
    
    # Count sums
    counts = {}
    
    # Optimization: Use local variables
    pl = powerful_list
    cnt = counts
    
    # We can also try to break early if time is up
    # But checking time in inner loop is slow.
    # We'll check every 10,000 iterations of i
    
    for i in range(n_list):
        a = pl[i]
        if i % 1000 == 0 and time.time() - start_time > timeout - 20:
            break
            
        for j in range(i, n_list):
            b = pl[j]
            s = a + b
            cnt[s] = cnt.get(s, 0) + 1
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Reconstruct pairs
    powerful_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
