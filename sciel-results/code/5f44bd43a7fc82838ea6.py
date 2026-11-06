"""Rewritten program: sparse powerful-number pair counting with larger L."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerfuls(limit):
    """Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    We iterate over square-free b and compute a^2 * b^3.
    """
    powerfuls = set()
    # b must be square-free and b^3 <= limit
    max_b = int(limit ** (1/3)) + 2
    # Precompute square-free numbers up to max_b
    # A number is square-free if no square of a prime divides it.
    # For small max_b (up to ~4642 for 10^8), we can check each.
    # Primes up to sqrt(max_b)
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    return powerfuls

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Generate powerful numbers up to L
    # The key insight from the improved solution: use sparse vectorized counting
    # instead of FFT, allowing L up to 10^8.
    L = 10**8
    
    powerfuls = _generate_powerfuls(L)
    powerfuls_sorted = np.array(sorted(powerfuls), dtype=np.int64)
    powerfuls_set = set(powerfuls)
    
    n = len(powerfuls_sorted)
    # print(f"Generated {n} powerful numbers up to {L}")
    
    # Time check
    if time.time() > deadline - 60:
        # Fallback: just return a simple answer
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Use numpy vectorized operations to count pairs.
    # For each a in powerfuls_sorted, compute b = N - a for candidate N values.
    # But we don't know N in advance.
    
    # Alternative: Compute all pairwise sums in a vectorized manner.
    # For n ~ 10000-20000, n^2 is 10^8 to 4*10^8. 
    # We can do this in chunks to avoid memory issues.
    
    # Actually, let's think differently. We want to find N that maximizes the count.
    # We can use a histogram approach:
    # For each a in powerfuls, for each b in powerfuls with b >= a, 
    # increment count[a+b].
    # This is O(n^2) but we can vectorize the inner loop.
    
    # Memory: we need a dict or array to hold counts.
    # Max sum is 2*L = 2*10^8. An array of that size is too big (200M * 4 bytes = 800MB).
    # Use a dict instead.
    
    # But O(n^2) with n=20000 is 4*10^8 operations. In Python, this is too slow.
    # In numpy, we can vectorize.
    
    # Let's try: for each a, compute all b >= a, then a + b.
    # This is n iterations, each doing a vectorized add of size ~n/2.
    # Total: n * (n/2) = n^2/2 operations, but vectorized.
    
    # Actually, a better approach:
    # Create the full pairwise sum matrix in chunks.
    # For chunk of size C, compute sums for a[i] + b[j] for i in chunk.
    
    # Let me try a different approach: use the fact that we want the MODE of the sum distribution.
    # We can sample or use a smarter algorithm.
    
    # Actually, let's just do the direct O(n^2) but with numpy vectorization in chunks.
    # n is probably around 10000-30000 for L=10^8.
    
    # Let's estimate n more carefully.
    # For L = 10^8:
    # b=1: a <= 10000, so 10000 values
    # b=2: 8a^2 <= 10^8 => a <= 3535, so 3535 values
    # b=3: 27a^2 <= 10^8 => a <= 1924, so 1924 values
    # b=5: 125a^2 <= 10^8 => a <= 894
    # b=6: 216a^2 <= 10^8 => a <= 680
    # ...
    # Total is roughly 10000 + 3535 + 1924 + 894 + 680 + ... ≈ 20000-25000
    
    # n^2/2 for n=25000 is ~3*10^8. This is too slow even vectorized in a loop.
    
    # Better approach: Use FFT after all, but with a smarter memory layout.
    # Or: use the sparse structure.
    
    # Actually, the key insight: we don't need ALL pairwise sums.
    # We just need to find the N with the most representations.
    # 
    # Let's use a probabilistic/heuristic approach:
    # 1. Randomly sample pairs and count frequencies.
    # 2. Use the most frequent N as a candidate.
    # 3. Verify by exact counting.
    # 4. Also check neighbors of the best candidate.
    
    # But this might miss the true optimum.
    
    # Let's try a hybrid: 
    # - For small n (< 5000), do full O(n^2) enumeration.
    # - For larger n, use a smarter method.
    
    if n <= 5000:
        # Full enumeration
        from collections import defaultdict
        count_dict = defaultdict(int)
        pair_dict = defaultdict(list)
        for i in range(n):
            a = powerfuls_sorted[i]
            # b must be >= a
            # Find start index for b >= a
            start_j = i  # since sorted, b[j] >= a for j >= i
            for j in range(start_j, n):
                b = powerfuls_sorted[j]
                s = a + b
                count_dict[s] += 1
                pair_dict[s].append([int(a), int(b)])
        
        # Find max
        best_N = max(count_dict, key=count_dict.get)
        best_pairs = pair_dict[best_N]
        
        if time.time() > deadline - 5:
            return {'N': int(best_N), 'pairs': best_pairs}
        
        return {'N': int(best_N), 'pairs': best_pairs}
    
    # For larger n, use chunked vectorized approach
    # Process in chunks to manage memory
    from collections import defaultdict
    count_dict = defaultdict(int)
    pair_dict = defaultdict(list)
    
    chunk_size = 200
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        a_chunk = powerfuls_sorted[i_start:i_end]
        
        # For each a in chunk, we need b >= a
        # We can vectorize: for each a, compute a + b for all b >= a
        # But b >= a means b is in powerfuls_sorted[j] where j >= i_start (approximately)
        
        # Actually, since a is increasing and b must be >= a, and both are sorted:
        # For a = powerfuls_sorted[i], b must be in powerfuls_sorted[j] where powerfuls_sorted[j] >= a
        # Since the array is sorted, this is j >= i (by index).
        
        # For the chunk, we can do:
        # For each row i in the chunk, compute a[i] + b[j] for j >= i
        
        # This is still O(chunk_size * n) per chunk, total O(n^2).
        # But vectorized per chunk.
        
        for i in range(i_start, i_end):
            a = powerfuls_sorted[i]
            # b values: powerfuls_sorted[i:] (all b >= a since sorted)
            b_vals = powerfuls_sorted[i:]
            sums = a + b_vals  # vectorized
            # Count
            # We need to update count_dict for each sum
            # This is still a loop over sums... 
            # Let's use numpy unique with counts
            unique_sums, counts = np.unique(sums, return_counts=True)
            for s, c in zip(unique_sums, counts):
                s = int(s)
                count_dict[s] += int(c)
                # We also need pairs, which is expensive to store
                # For now, let's just track counts, then re-extract pairs for the best N
        
        # Time check
        if time.time() > deadline - 30:
            break
    
    # Find best N from count_dict
    if not count_dict:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(count_dict, key=count_dict.get)
    best_count = count_dict[best_N]
    
    # Now extract pairs for best_N
    pairs = []
    for a in powerfuls_sorted:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([int(a), int(b)])
    
    # Also check a few neighbors of best_N in case of ties or near-misses
    # And check if any neighbor has more pairs
    best_final_N = best_N
    best_final_pairs = pairs
    best_final_count = len(pairs)
    
    # Check top 20 candidates
    top_candidates = sorted(count_dict.items(), key=lambda x: -x[1])[:20]
    for cand_N, cand_count in top_candidates:
        if int(cand_N) < 2:
            continue
        cand_pairs = []
        for a in powerfuls_sorted:
            if a > cand_N // 2:
                break
            b = int(cand_N) - a
            if b in powerfuls_set:
                cand_pairs.append([int(a), int(b)])
        if len(cand_pairs) > best_final_count:
            best_final_count = len(cand_pairs)
            best_final_N = int(cand_N)
            best_final_pairs = cand_pairs
    
    if time.time() > deadline - 5:
        return {'N': best_final_N, 'pairs': best_final_pairs}
    
    return {'N': best_final_N, 'pairs': best_final_pairs}

# EVOLVE-BLOCK-END
