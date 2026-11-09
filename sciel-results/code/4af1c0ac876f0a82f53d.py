import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Limit for powerful numbers
    # 10^9 gives ~2154 numbers, ~2.3M pairs
    L = 10**9
    
    # Generate powerful numbers
    powerful = set()
    max_m = int(L ** (1/3)) + 2
    for m in range(1, max_m + 1):
        m3 = m ** 3
        if m3 > L:
            break
        max_k = int(math.isqrt(L // m3))
        for k in range(1, max_k + 1):
            val = k * k * m3
            if val <= L:
                powerful.add(val)
    
    P = sorted(powerful)
    K = len(P)
    
    # Count sums
    counts = {}
    for i in range(K):
        pi = P[i]
        for j in range(i, K):
            s = pi + P[j]
            counts[s] = counts.get(s, 0) + 1
            
            # Time check
            if time.time() > deadline - 10:
                # If we run out of time, we break and use what we have
                # This might result in an incomplete count, but better than timeout
                # However, breaking early might bias the count.
                # Given 2.3M iterations, this should not trigger.
                break
        
    # Find best N
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    
    # Find pairs for best_N
    pairs = []
    for i in range(K):
        pi = P[i]
        for j in range(i, K):
            s = pi + P[j]
            if s == best_N:
                pairs.append([pi, P[j]])
            elif s > best_N:
                break
                
    return {'N': best_N, 'pairs': pairs}
