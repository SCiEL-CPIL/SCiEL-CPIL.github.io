import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    LIMIT = 10**12
    
    MAX_B = 10000
    is_squarefree = [True] * (MAX_B + 1)
    for i in range(2, isqrt(MAX_B) + 1):
        sq = i * i
        for j in range(sq, MAX_B + 1, sq):
            is_squarefree[j] = False
    
    powerful_set = set()
    for b in range(1, MAX_B + 1):
        if not is_squarefree[b]:
            continue
        b3 = b ** 3
        if b3 > LIMIT:
            break
        max_a = isqrt(LIMIT // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            powerful_set.add(val)
    
    powerful_sorted = sorted(powerful_set)
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Generate candidate N values
    candidates = set()
    
    # Around twice the median powerful number
    if powerful_sorted:
        mid_idx = len(powerful_sorted) // 2
        mid_val = powerful_sorted[mid_idx]
        for delta in range(-200000, 200001, 500):
            N = 2 * mid_val + delta
            if 2 <= N <= LIMIT:
                candidates.add(N)
    
    # Logarithmically spaced candidates
    for m in range(4, 13):
        base = 10 ** m
        for k in range(1, 25):
            N = k * base
            if 2 <= N <= LIMIT:
                candidates.add(N)
    
    # Evaluate candidates
    for N in sorted(candidates):
        if time.time() > deadline:
            break
        count = 0
        pairs = []
        for a in powerful_sorted:
            if 2 * a > N:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    if best_N is None:
        best_N = 5
        best_pairs = [[1, 4]]
    
    return {'N': best_N, 'pairs': best_pairs}
