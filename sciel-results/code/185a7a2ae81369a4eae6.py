import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Set limit
    # Based on analysis, 5e6 should be safe in ~40-60s
    M = 5000000 
    
    # 1. Generate powerful numbers up to M
    powerful_set = set()
    # v^3 <= M => v <= M^(1/3)
    max_v = int(M ** (1/3)) + 1
    
    # We need v to be square-free?
    # A number n is powerful iff n = u^2 v^3 where v is square-free.
    # Yes, this representation is unique if v is square-free.
    # So we iterate square-free v.
    
    def is_square_free(n):
        if n < 1: return False
        if n == 1: return True
        # Check primes up to sqrt(n)
        # Since max_v is small (~170), we can just check divisibility by p^2
        # Primes up to 170: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173
        # Actually, we only need to check p^2 <= n.
        # For n up to 170, p^2 <= 170 => p <= 13.
        # Primes: 2, 3, 5, 7, 11, 13.
        for p in [2, 3, 5, 7, 11, 13]:
            if n % (p*p) == 0:
                return False
        return True

    for v in range(1, max_v + 1):
        if not is_square_free(v):
            continue
        v3 = v ** 3
        if v3 > M:
            break
        # u^2 * v^3 <= M => u^2 <= M / v^3 => u <= sqrt(M / v^3)
        max_u = isqrt(M // v3)
        for u in range(1, max_u + 1):
            powerful_set.add(u * u * v3)
            
    P = sorted(powerful_set)
    K = len(P)
    
    # Check time
    if time.time() > deadline:
         # Fallback to smaller set if generation was too slow (unlikely)
         return {'N': 2, 'pairs': [[1, 1]]}

    # 2. Count pairs
    # Max sum is 2*M
    # List of zeros
    counts = [0] * (2 * M + 1)
    
    # Optimization: local variables
    P_local = P
    K_local = K
    counts_local = counts
    
    # Double loop
    # We can potentially break if time is running out, but we need complete counts for correctness of the "max".
    # If we break, the max might be wrong.
    # So we should ensure M is small enough to finish.
    
    for i in range(K_local):
        a = P_local[i]
        # b must be >= a to keep pairs distinct (a<=b)
        # But for counting representations N=a+b, the order doesn't matter for the sum value.
        # However, the problem asks for "distinct pairs a <= b".
        # So we only count (a,b) where a <= b.
        # This is handled by j starting from i.
        
        # Inner loop
        # To speed up, we can use a list slice? No, addition is element-wise.
        # Just a loop.
        for j in range(i, K_local):
            counts_local[a + P_local[j]] += 1
            
    # Find max
    max_count = 0
    best_N = 0
    for n, c in enumerate(counts_local):
        if c > max_count:
            max_count = c
            best_N = n
            
    # Reconstruct pairs
    pairs = []
    # We need a <= b and a+b = best_N
    # And a, b in powerful_set
    P_set = set(P) # For O(1) lookup? We already have P sorted.
    # Binary search or just iterate P
    # Since P is sorted, we can do two pointers or binary search.
    # But K is small enough to just iterate.
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in P_set: # Using set for speed
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
