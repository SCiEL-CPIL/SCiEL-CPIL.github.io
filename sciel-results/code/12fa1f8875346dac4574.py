import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Limit for powerful number generation
    # 10^13 gives ~6 million powerful numbers, fits in memory
    limit = 10**13
    
    # Generate powerful numbers
    # Powerful numbers are a^2 * b^3 where b is square-free
    powerful = set([1])
    
    # Max b such that b^3 <= limit
    max_b = int(limit ** (1/3)) + 1
    
    # Precompute square-free check? 
    # b is up to 21544. 
    # Checking square-free for each b is fast.
    
    def is_square_free(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        b_cubed = b ** 3
        # a^2 <= limit / b^3
        max_a = isqrt(limit // b_cubed)
        for a in range(1, max_a + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    powerful_list = sorted(powerful)
    powerful_set = powerful  # Use set for O(1) lookup
    
    def count_reps(N):
        count = 0
        pairs = []
        # Iterate through a
        # a <= N/2
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        return count, pairs

    best_count = 0
    best_N = 0
    best_pairs = []
    
    # Candidate N's
    candidates = set()
    
    # Add some structured candidates
    for i in range(1, 100):
        candidates.add(limit // i)
        
    # Add random candidates
    # We can afford to test ~500-1000 candidates
    # Each test takes O(len(powerful_list)) ~ 6e6 ops.
    # 6e6 ops ~ 0.05s in Python? Maybe 0.1s.
    # 100 candidates ~ 10s.
    # We have ~100s. So 1000 candidates is doable.
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        # Sample from upper half to maximize N
        N = random.randint(limit // 2, limit)
        candidates.add(N)
        
    # Also try some specific small N's just in case? No, we want max score.
    # Large N's have higher potential for absolute count?
    # Actually, count is roughly constant in expectation.
    # But variance is high.
    
    for N in candidates:
        if time.time() > deadline:
            break
        c, p = count_reps(N)
        if c > best_count:
            best_count = c
            best_N = N
            best_pairs = p
            
    # If we have time, do more random search
    while time.time() < deadline - 5:
        N = random.randint(limit // 2, limit)
        if N in candidates:
            continue
        c, p = count_reps(N)
        if c > best_count:
            best_count = c
            best_N = N
            best_pairs = p
        candidates.add(N)
        
    return {'N': best_N, 'pairs': best_pairs}
