import time
from math import isqrt
from collections import Counter
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    MAX_N = 10**18
    
    # Generate all powerful numbers up to 10^18
    # n = a^2 * b^3 where a, b >= 1
    powerful = set()
    max_b = int(MAX_N ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > MAX_N:
            break
        max_a = isqrt(MAX_N // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n > MAX_N:
                break
            powerful.add(n)
    
    powerful_list = sorted(powerful)
    powerful_set = set(powerful_list)
    
    # Generate candidate N values
    # 1. From sums of the smallest powerful numbers
    K = min(1500, len(powerful_list))
    small_powerful = powerful_list[:K]
    candidate_counter = Counter()
    for i in range(K):
        a = small_powerful[i]
        for j in range(i, K):
            b = small_powerful[j]
            n = a + b
            if 2 <= n <= MAX_N:
                candidate_counter[n] += 1
    
    # Take top candidates by frequency in small sums
    top_from_small = [n for n, c in candidate_counter.most_common(150)]
    
    candidates = set(top_from_small)
    
    # 2. Add candidates around powers of 10 and other structured values
    for i in range(2, 19):
        base = 10**i
        for delta in range(-200, 201, 20):
            n = base + delta
            if 2 <= n <= MAX_N:
                candidates.add(n)
    
    # 3. Add random candidates
    random.seed(42)
    for _ in range(100):
        candidates.add(random.randint(2, MAX_N))
    
    # Count representations for each candidate
    best_N = 2
    best_count = 0
    best_pairs = []
    
    # Process candidates, prioritizing those with high small-sum counts
    candidate_list = list(candidates)
    
    for N in candidate_list:
        if time.time() > deadline:
            break
        if N < 2 or N > MAX_N:
            continue
        
        pairs = []
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
    
    return {'N': best_N, 'pairs': best_pairs}
