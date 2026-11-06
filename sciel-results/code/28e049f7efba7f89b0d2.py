import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate powerful numbers up to L
    L = 10**7
    
    pows = set([1])
    max_i = math.isqrt(L)
    
    # Precompute cubes
    max_j = int(round(L ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j + 1)]
    
    for i in range(1, max_i + 1):
        sq = i * i
        if sq > L:
            break
        max_j_local = int(round((L // sq) ** (1/3)))
        if max_j_local > max_j:
            max_j_local = max_j
        for j in range(1, max_j_local + 1):
            val = sq * cubes[j-1]
            if val <= L:
                pows.add(val)
            else:
                break
        if time.time() > deadline - 50:
            break

    P_list = sorted(pows)
    P_set = set(P_list)
    K = len(P_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Direct counting of sums using dictionary
    sum_counts = defaultdict(int)
    
    for i in range(K):
        a = P_list[i]
        for j in range(i, K):
            b = P_list[j]
            s = a + b
            if s <= 10**18:
                sum_counts[s] += 1
        if time.time() > deadline - 40:
            break
    
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the sum with maximum count
    best_N = max(sum_counts, key=sum_counts.get)
    
    # Verify and collect pairs
    pairs = []
    for a in P_list:
        if a > best_N:
            break
        b = best_N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([int(a), int(b)])
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': int(best_N), 'pairs': pairs}
