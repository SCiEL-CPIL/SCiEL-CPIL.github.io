# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Generate powerful numbers up to LIMIT
    # We choose LIMIT = 10^9 to keep M ~ 20000, so M^2 ~ 4e8 operations
    LIMIT = 10**9
    
    powerful = set()
    # v^3 <= LIMIT => v <= (10^9)^(1/3) = 1000
    max_v = int(LIMIT ** (1/3)) + 1
    
    for v in range(1, max_v + 1):
        v3 = v ** 3
        if v3 > LIMIT:
            break
        max_u = isqrt(LIMIT // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= LIMIT:
                powerful.add(val)
            else:
                break
        
        if time.time() > deadline:
            break
    
    P = sorted(powerful)
    M = len(P)
    
    # Count sums
    # We want to find N with max count of pairs (a, b) with a <= b, a+b=N
    # Using a dictionary to count occurrences of sums
    counts = {}
    best_N = -1
    best_count = 0
    
    # Optimization: If M is too large, we might not finish.
    # But for 10^9, M~20k, M^2/2 ~ 2e8.
    # Python dict update is ~100ns? 2e8 * 100ns = 20s.
    # This should fit.
    
    for i in range(M):
        a = P[i]
        # b must be >= a
        # We can break early if a + P[j] > 10**18, but LIMIT is 10^9, so sum < 2e9
        for j in range(i, M):
            b = P[j]
            s = a + b
            # Check deadline periodically? 
            # Checking time.time() in inner loop is expensive.
            # We can check every 10000 iterations.
            if j % 10000 == 0:
                if time.time() > deadline:
                    break
            c = counts.get(s, 0) + 1
            counts[s] = c
            if c > best_count:
                best_count = c
                best_N = s
        
        if time.time() > deadline:
            break
    
    if best_N == -1:
        # Fallback if something went wrong
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Reconstruct pairs for best_N
    pairs = []
    members = set(P)
    for a in P:
        if a > best_N:
            break
        b = best_N - a
        if b >= a and b in members:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
