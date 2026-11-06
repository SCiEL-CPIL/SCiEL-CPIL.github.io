import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Use a limit that ensures K is small enough for O(K^2) to be fast in NumPy
    # K ~ sqrt(L) * constant. For L=10^8, K~10^4, K^2=10^8.
    # NumPy can handle 10^8 elements in < 5s typically.
    limit = 100000000
    
    # Generate powerful numbers
    pows = set([1])
    max_i = math.isqrt(limit)
    # j^3 <= limit => j <= limit^(1/3)
    max_j_global = int(round(limit ** (1/3))) + 2
    cubes = [j ** 3 for j in range(1, max_j_global + 1)]
    
    for i in range(1, max_i + 1):
        sq = i * i
        if sq > limit:
            break
        # i^2 * j^3 <= limit => j^3 <= limit / i^2
        max_j = int(round((limit // sq) ** (1/3)))
        if max_j > max_j_global:
            max_j = max_j_global
        for j in range(1, max_j + 1):
            val = sq * cubes[j-1]
            if val <= limit:
                pows.add(val)
            else:
                break
        if time.time() > deadline - 30: # Leave time for NumPy
             break

    P = np.array(sorted(list(pows)), dtype=np.int64)
    K = len(P)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Compute all sums P_i + P_j
    # This creates a K x K matrix
    # For K=10000, this is 10^8 elements (800 MB).
    # If memory is a concern, we could block it, but let's try full first.
    # If it fails, we'd need to reduce limit.
    
    sums = P[:, None] + P[None, :]
    
    # Count occurrences of each sum
    unique_sums, counts = np.unique(sums, return_counts=True)
    
    # Find the sum with the maximum count
    max_c = np.max(counts)
    idx = np.argmax(counts)
    best_sum = unique_sums[idx]
    
    # Calculate number of distinct pairs a <= b
    # C_full = 2 * (pairs with a < b) + (1 if a == b exists)
    # M = (C_full + D) / 2
    D = 0
    if best_sum % 2 == 0:
        half = best_sum // 2
        # Check if half is in P
        # P is sorted, use searchsorted
        pos = np.searchsorted(P, half)
        if pos < K and P[pos] == half:
            D = 1
            
    actual_count = (max_c + D) // 2
    
    # Retrieve the actual pairs to verify and return
    pows_set = set(P.tolist())
    pairs = []
    # Iterate through P to find pairs summing to best_sum
    # Since P is sorted, we can break early
    for a in P:
        if a > best_sum:
            break
        b = best_sum - a
        if b < a:
            continue # We only want a <= b
        if b in pows_set:
            pairs.append([int(a), int(b)])
            
    # If no pairs found (should not happen if count > 0), fallback
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': int(best_sum), 'pairs': pairs}
