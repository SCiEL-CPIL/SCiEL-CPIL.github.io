import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate powerful numbers up to M
    def generate_powerful(M):
        pw = set()
        # a = u^2 v^3
        max_v = int(M ** (1/3)) + 2
        for v in range(1, max_v + 1):
            v3 = v ** 3
            if v3 > M:
                break
            max_u = math.isqrt(M // v3)
            for u in range(1, max_u + 1):
                a = u * u * v3
                if a <= M:
                    pw.add(a)
        return sorted(pw)
    
    # Strategy: Exact pair enumeration for small M, then lift
    # M = 10^7 -> K ~ 10k. K^2 = 100M operations.
    # Chunked bincount to manage memory.
    
    M = 10_000_000
    P = generate_powerful(M)
    K = len(P)
    
    if time.time() > deadline - 20:
        # Fallback to simple result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # We want to count pairs (a, b) with a <= b.
    # Using bincount on all pairs (a, b) will count both (a,b) and (b,a).
    # For a != b, count is doubled. For a == b, count is 1.
    # We can just find the N with max total count, then verify/adjust.
    # Or simply count all pairs and divide by 2 (approx).
    # Better: just find the N with max frequency in the sum array.
    
    max_sum = 2 * M
    counts = np.zeros(max_sum + 1, dtype=np.int64)
    
    # Process in chunks to avoid creating a huge KxK array at once
    chunk_size = 500
    P_np = np.array(P, dtype=np.int64)
    
    for i in range(0, K, chunk_size):
        end_i = min(i + chunk_size, K)
        # Sums of P[i:end_i] with all P
        # Shape: (chunk_size, K)
        sums = P_np[i:end_i, None] + P_np[None, :]
        # Flatten and bincount
        # minlength ensures the array size is correct
        chunk_counts = np.bincount(sums.ravel(), minlength=max_sum + 1)
        counts += chunk_counts
        
        if time.time() > deadline - 10:
            break
    
    if time.time() > deadline - 5:
        # Return best found so far (might be incomplete)
        # Find max
        max_idx = np.argmax(counts)
        N_cand = int(max_idx)
        # Get pairs
        P_set = set(P)
        pairs = []
        for a in P:
            if a > N_cand // 2:
                break
            b = N_cand - a
            if b in P_set:
                pairs.append([a, b])
        return {'N': N_cand, 'pairs': pairs}

    # Find the N with the highest count
    # Note: counts[N] includes (a,b) and (b,a).
    # The actual number of distinct pairs a<=b is roughly counts[N]/2.
    # We want to maximize this.
    max_idx = np.argmax(counts)
    N_small = int(max_idx)
    
    # Verify and get pairs for N_small
    P_set = set(P)
    pairs_small = []
    for a in P:
        if a > N_small // 2:
            break
        b = N_small - a
        if b in P_set:
            pairs_small.append([a, b])
    
    # Lift N_small to N_final
    # N_final = N_small * k^2 <= 10^18
    max_N = 10**18
    # k = floor(sqrt(max_N / N_small))
    k = int(math.isqrt(max_N // N_small))
    
    if k == 0:
        # Should not happen since N_small >= 2
        k = 1
        
    N_final = N_small * k * k
    
    # Scale pairs
    pairs_final = []
    for a, b in pairs_small:
        a_new = a * k * k
        b_new = b * k * k
        pairs_final.append([a_new, b_new])
        
    # Check if we have time to search for extra pairs in N_final?
    # Probably not, as checking powerfulness for large numbers is slow.
    # But we can check a few small a's.
    # However, the score is the number of pairs returned.
    # We return the scaled pairs.
    
    return {'N': N_final, 'pairs': pairs_final}
