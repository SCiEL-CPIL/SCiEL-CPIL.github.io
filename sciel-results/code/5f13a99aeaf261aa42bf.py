import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # 1. Generate powerful numbers up to a limit M
    # We want M as large as possible to find N with many representations.
    # Powerful numbers are of the form u^2 * v^3 where v is square-free.
    # Density is roughly N^(-1/3).
    # Let's aim for M = 200,000,000 (2e8).
    # Count of powerful numbers up to 2e8 is approx (2e8)^(2/3) * C.
    # (2e8)^(1/3) is approx 585.
    # Number of square-free v up to 585 is approx 585 * 6/pi^2 ~ 350.
    # For each v, u goes up to sqrt(M/v^3).
    # Total count is roughly sum_{v squarefree} sqrt(M/v^3) = sqrt(M) * sum 1/v^(3/2) ~ sqrt(M) * 1.5.
    # sqrt(2e8) ~ 14142. So count ~ 21000.
    # Wait, the constant is larger.
    # Let's just generate and see.
    
    M = 200_000_000
    
    # Check time
    if time.time() - start_time > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Generate powerful numbers
    powerful_list = []
    
    # We need square-free v up to M^(1/3)
    max_v = int(M ** (1/3)) + 2
    
    # Sieve for square-free numbers? Or just check?
    # max_v is ~ 585. Checking each is cheap.
    def is_square_free(n):
        if n < 1: return False
        if n == 1: return True
        # Primes up to sqrt(585) ~ 24.
        # 2, 3, 5, 7, 11, 13, 17, 19, 23
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if n % (p*p) == 0:
                return False
        return True

    # Precompute square-free v
    square_free_vs = []
    for v in range(1, max_v + 1):
        if is_square_free(v):
            square_free_vs.append(v)
            
    for v in square_free_vs:
        v3 = v ** 3
        if v3 > M:
            continue
        max_u = isqrt(M // v3)
        if max_u == 0:
            continue
        # Generate u^2 * v3 for u in 1..max_u
        # u^2 is vectorizable
        us = np.arange(1, max_u + 1)
        pows = (us * us) * v3
        powerful_list.append(pows)
        
    if time.time() - start_time > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Concatenate and unique
    if not powerful_list:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    all_pows = np.concatenate(powerful_list)
    all_pows = np.unique(all_pows)
    
    # Filter <= M (should be already, but just in case)
    all_pows = all_pows[all_pows <= M]
    
    K = len(all_pows)
    
    if time.time() - start_time > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}

    # 2. Count pairs using batched numpy operations
    # We want to count how many pairs (a, b) from all_pows sum to N.
    # Since all_pows is sorted, we can do:
    # For each a in all_pows, b = N - a.
    # But we don't know N.
    # Strategy: Compute all sums a + b where a <= b?
    # That's O(K^2) memory if we do it all at once. K ~ 20000. K^2 = 4e8. Too big for RAM (3.2 GB).
    # We need to batch.
    
    # We can iterate over i (index of a) in batches, and for each batch, compute sums with all j >= i?
    # Or simpler:
    # Use a dictionary or array for counts.
    # Max sum is 2*M = 4e8. An array of size 4e8 is 3.2 GB (int32). Too big.
    # We need a sparse approach or a smaller M.
    
    # Let's reduce M to 50,000,000 (5e7).
    # K ~ sqrt(5e7) * 1.5 ~ 7000 * 1.5 ~ 10000.
    # Max sum 1e8. Array size 1e8 is 400 MB (int32). Feasible.
    
    # Let's re-evaluate M.
    # If M = 50,000,000:
    # K ~ 10,000.
    # Array size 100,000,001.
    # Batch processing:
    # For i in range(0, K, batch_size):
    #   a_batch = all_pows[i : i+batch_size]
    #   # We want sums a + b where b >= a?
    #   # Actually, for counting representations N = a+b, order doesn't matter.
    #   # But we want to count distinct pairs {a,b}.
    #   # If we count all pairs (a,b) with a,b in P, we count (a,b) and (b,a).
    #   # If a != b, count is 2x. If a == b, count is 1.
    #   # We can just count all pairs (a,b) and divide? No, because we want distinct pairs a<=b.
    #   # It's easier to just enforce a <= b in the generation.
    
    # Let's try M = 100,000,000 (1e8).
    # K ~ 15,000.
    # Max sum 2e8. Array 2e8 * 4 bytes = 800 MB. Might be tight but okay.
    # Let's stick with M = 50,000,000 to be safe on memory and time.
    # Actually, the previous program used M=5e6 and scored 0.06.
    # The inspiration suggests 100M limit with numpy batching improved score to 5.9.
    # So let's aim for M = 100,000,000.
    
    # Re-generate for M = 100,000,000
    M = 100_000_000
    powerful_list = []
    max_v = int(M ** (1/3)) + 2
    square_free_vs = []
    for v in range(1, max_v + 1):
        if is_square_free(v):
            square_free_vs.append(v)
            
    for v in square_free_vs:
        v3 = v ** 3
        if v3 > M:
            continue
        max_u = isqrt(M // v3)
        if max_u == 0:
            continue
        us = np.arange(1, max_u + 1)
        pows = (us * us) * v3
        powerful_list.append(pows)
        
    if time.time() - start_time > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}

    all_pows = np.concatenate(powerful_list)
    all_pows = np.unique(all_pows)
    all_pows = all_pows[all_pows <= M]
    K = len(all_pows)
    
    if time.time() - start_time > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Max sum is 2*M
    max_sum = 2 * M
    # Use int32 for counts? Max count is K ~ 15000. int32 is fine.
    # Array size 200,000,001. 800 MB.
    # Let's check if we have enough memory. Usually yes.
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Batch processing
    # We want to count pairs (a, b) with a <= b.
    # For each i, a = all_pows[i].
    # b must be >= a.
    # So b is in all_pows[j] where j >= i.
    # We can't easily vectorize the "j >= i" condition for all i at once in a simple batch.
    # Alternative: Count ALL pairs (a, b) and handle the a <= b constraint later?
    # If we count all pairs, counts[N] will be the number of ordered pairs.
    # For N, let C be the count.
    # Let D be the number of pairs with a=b (i.e., 2a=N, so N even and N/2 in P).
    # Then number of distinct pairs {a,b} with a<=b is (C - D) / 2 + D = (C + D) / 2.
    # So we can just count all ordered pairs.
    
    # Counting all ordered pairs:
    # For each batch of a's, add all_pows to each a.
    # sums = a_batch[:, None] + all_pows[None, :]
    # This is batch_size * K matrix.
    # If K=15000, batch_size=1000, matrix is 1000x15000 = 15e6 elements.
    # 15e6 * 4 bytes (int32) = 60 MB. Fine.
    # We can use np.bincount on the flattened sums.
    
    batch_size = 1000
    for i in range(0, K, batch_size):
        end = min(i + batch_size, K)
        a_batch = all_pows[i:end]
        # Compute all sums
        sums = a_batch[:, None] + all_pows[None, :]
        # Flatten and bincount
        # sums are up to 2*M
        # bincount requires non-negative integers
        flat_sums = sums.ravel()
        # We need to add to counts
        # np.add.at is slow.
        # We can create a temporary array and add?
        # Or use np.bincount and then add to counts.
        # bincount returns an array of length max(flat_sums)+1.
        # This might be large (2*M).
        # But we can just do:
        # temp_counts = np.bincount(flat_sums, minlength=max_sum+1)
        # counts += temp_counts
        # This creates a large temporary array each batch.
        # 200 batches * 800 MB = 160 GB of memory traffic. Might be slow.
        
        # Alternative:
        # Use a dictionary? No, numpy is faster for dense.
        # Let's try the bincount approach first.
        
        temp_counts = np.bincount(flat_sums, minlength=max_sum + 1)
        counts += temp_counts
        
        if time.time() - start_time > deadline:
            # Return best found so far?
            # We need to find max in counts so far.
            # But counts is partial.
            # Just return a valid result.
            break

    if time.time() - start_time > deadline:
        # Find max in current counts
        max_count = np.max(counts)
        best_N = np.argmax(counts)
        # Reconstruct pairs
        # This might be wrong if we broke early, but it's a valid N.
        # However, the count might be low.
        # Let's just return N=2 to be safe if we run out of time?
        # No, return the best found.
        pass

    # Find max count
    max_count = np.max(counts)
    best_N = np.argmax(counts)
    
    # Reconstruct pairs
    # We need distinct pairs a <= b such that a + b = best_N and a, b are powerful.
    # We have all_pows sorted.
    # We can use two pointers or binary search.
    # Since all_pows is sorted, for each a in all_pows:
    #   if a > best_N // 2: break
    #   b = best_N - a
    #   if b in powerful_set: add pair
    
    powerful_set = set(all_pows.tolist())
    pairs = []
    for a in all_pows:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([int(a), int(b)])
            
    return {'N': int(best_N), 'pairs': pairs}
