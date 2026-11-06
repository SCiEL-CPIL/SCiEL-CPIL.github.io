import time
from math import lcm
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    weights = [L // d for d in denominators]
    
    mid = n // 2
    weights_1 = weights[:mid]
    weights_2 = weights[mid:]
    
    def gen_sums(weights_list):
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            bit = 1 << i
            new_sums = []
            for s, m in sums:
                ns = s + w
                if ns < L:
                    new_sums.append((ns, m | bit))
            sums.extend(new_sums)
            if time.time() > deadline:
                break
        return sums

    sums_1 = gen_sums(weights_1)
    sums_2 = gen_sums(weights_2)
    
    sums_2.sort(key=lambda x: x[0])
    sums_2_values = [s[0] for s in sums_2]
    
    best_sum = -1
    best_mask = 0
    
    for s1, m1 in sums_1:
        if time.time() > deadline:
            break
        target = L - s1 - 1
        idx = bisect.bisect_right(sums_2_values, target)
        if idx > 0:
            s2, m2 = sums_2[idx-1]
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask = m1 | (m2 << mid)
                
    if best_mask == 0:
        return [0]
        
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(i)
            
    return result_indices
