"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using square-free sieve."""
    max_b = int(limit ** (1.0 / 3.0)) + 1
    
    is_sq_free = [True] * (max_b + 1)
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
    is_sq_free[0] = False
    
    powerful_set = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful_set.add(val)
    
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Use a larger limit to get more powerful numbers
    limit = 50_000_000
    powerful = _generate_powerful(limit)
    members = set(powerful)
    n_pow = len(powerful)
    
    if time.time() > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Strategy: Use a dictionary to count representations for each N
    # Process powerful numbers in chunks, and for each chunk, compute sums
    # with all powerful numbers >= current value
    
    # Key optimization: only consider pairs where a <= b, so for index i,
    # we only sum with indices j >= i
    
    # Use a more efficient approach: for small powerful numbers (which are
    # more numerous), they participate in many sums. Process them first.
    
    # Sort powerful numbers (already sorted)
    # For efficiency, use numpy for vectorized operations on chunks
    
    import numpy as np
    arr = np.array(powerful, dtype=np.int64)
    
    # Count representations using a dictionary
    # Process in batches to manage memory and time
    counts = {}
    
    # Process each powerful number as 'a' and find all 'b' >= 'a' such that a+b is counted
    # But this is O(n^2). Instead, use a smarter approach:
    
    # For each possible sum N, the number of representations is the number of
    # powerful numbers a such that a <= N/2 and N-a is also powerful.
    
    # Alternative: Use convolution-like approach or direct counting with early termination
    
    # Let's use a hybrid: for the first K powerful numbers (smaller ones),
    # compute all their sums with all powerful numbers, and count.
    # Small powerful numbers are more likely to be part of many representations.
    
    # Determine how many we can afford to process
    # Each pair (i, j) with i <= j contributes to count of arr[i] + arr[j]
    # Total pairs ~ n_pow^2 / 2
    
    # For n_pow ~ 20000, that's ~200M pairs - too many to do naively
    
    # Better approach: Use the fact that we only need to find the MAX count N
    # and collect its pairs. We can use a randomized or heuristic approach:
    
    # 1. Sample some candidate N values from sums of small powerful numbers
    # 2. For each candidate, count how many pairs sum to it
    # 3. Keep the best
    
    # Generate candidate N values from sums of small powerful numbers
    small_k = min(2000, n_pow)
    small_arr = arr[:small_k]
    
    # Generate candidate sums from small-small pairs
    candidates = set()
    for i in range(small_k):
        for j in range(i, small_k):
            s = small_arr[i] + small_arr[j]
            candidates.add(int(s))
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break
    
    # Also add some random candidates from larger sums
    if len(candidates) < 10000 and time.time() < deadline - 20:
        import random
        for _ in range(10000):
            i = random.randint(0, n_pow - 1)
            j = random.randint(i, n_pow - 1)
            candidates.add(int(arr[i] + arr[j]))
    
    # For each candidate N, count representations
    best_N = 2
    best_count = 1  # [1,1] is always valid
    best_pairs = [[1, 1]]
    
    for N in candidates:
        if time.time() > deadline - 5:
            break
        
        # Count pairs (a, b) with a <= b, a + b = N, both powerful
        count = 0
        pairs = []
        for a in powerful:
            if 2 * a > N:
                break
            b = N - a
            if b in members:
                count += 1
                pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    # If we didn't find anything better, try a direct approach with smaller limit
    if best_count <= 2 and time.time() < deadline - 10:
        # Try with a smaller, denser set
        limit2 = 5_000_000
        powerful2 = _generate_powerful(limit2)
        members2 = set(powerful2)
        arr2 = np.array(powerful2, dtype=np.int64)
        n_pow2 = len(powerful2)
        
        counts2 = {}
        for i in range(n_pow2):
            a = arr2[i]
            sums = a + arr2[i:]
            unique_sums, cnts = np.unique(sums, return_counts=True)
            for s, c in zip(unique_sums, cnts):
                s_int = int(s)
                counts2[s_int] = counts2.get(s_int, 0) + int(c)
        
        if counts2:
            best_N2 = max(counts2, key=counts2.get)
            if counts2[best_N2] > best_count:
                best_N = best_N2
                best_count = counts2[best_N2]
                best_pairs = []
                for a in powerful2:
                    if 2 * a > best_N:
                        break
                    b = best_N - a
                    if b in members2:
                        best_pairs.append([a, b])
    
    best_pairs = sorted(set(tuple(p) for p in best_pairs))
    best_pairs = [list(p) for p in best_pairs]
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
