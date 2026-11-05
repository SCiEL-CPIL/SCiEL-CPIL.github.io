import time
from math import isqrt

def construct():
    start_time = time.time()
    timeout = 110
    
    # M = 5 * 10^8
    # K ~ sqrt(5*10^8) ~ 22360
    # Pairs ~ 2.5 * 10^8
    M = 500_000_000

    powerful = set()
    max_y = int(M ** (1/3)) + 1

    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            powerful.add(p)
            
    powerful_list = sorted(powerful)
    n_list = len(powerful_list)
    p_list = powerful_list
    
    counts = {}
    
    if time.time() - start_time > timeout - 10:
        return {'N': 2, 'pairs': [[1, 1]]}

    for i in range(n_list):
        a = p_list[i]
        if i % 100 == 0 and time.time() - start_time > timeout - 5:
            break
        for j in range(i, n_list):
            s = a + p_list[j]
            counts[s] = counts.get(s, 0) + 1
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    
    pairs = []
    p_set = set(p_list)
    for a in p_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
