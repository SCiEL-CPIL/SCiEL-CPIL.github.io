import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Limit for powerful number generation
    # Increased limit to increase expected number of representations
    limit = 3 * 10**13
    
    # Generate powerful numbers
    powerful = set([1])
    
    # Max b such that b^3 <= limit
    max_b = int(limit ** (1/3)) + 1
    
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
        max_a = isqrt(limit // b_cubed)
        for a in range(1, max_a + 1):
            val = a * a * b_cubed
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    powerful_list = sorted(powerful)
    powerful_set = powerful
    
    def count_reps(N):
        count = 0
        pairs = []
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
    
    candidates = set()
    
    # Add candidates from random pair sums (biased towards high density)
    # We generate a few random pairs to find "hot" sums
    num_pair_samples = 500
    for _ in range(num_pair_samples):
        a = random.choice(powerful_list)
        b = random.choice(powerful_list)
        candidates.add(a + b)
        
    # Add random N candidates
    num_random_samples = 1000
    for _ in range(num_random_samples):
        # N should be such that a, b <= limit
        # So N <= 2 * limit
        # But we want a, b to be in the "bulk" of powerful numbers
        # So N in [limit, 2*limit] is good
        N = random.randint(limit, 2 * limit)
        candidates.add(N)
        
    # Check candidates
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
        N = random.randint(limit, 2 * limit)
        if N in candidates:
            continue
        c, p = count_reps(N)
        if c > best_count:
            best_count = c
            best_N = N
            best_pairs = p
        candidates.add(N)
        
    return {'N': best_N, 'pairs': best_pairs}
