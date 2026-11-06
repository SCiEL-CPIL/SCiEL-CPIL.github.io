import time
from math import isqrt
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using square-free sieve."""
    # b is the cube-free part? No, powerful number is a^2 * b^3 where b is square-free.
    # Actually, standard generation:
    # n is powerful iff for every prime p dividing n, p^2 divides n.
    # n = a^2 * b^3 where b is square-free.
    
    max_b = int(limit ** (1.0 / 3.0)) + 1
    
    # Sieve for square-free numbers up to max_b
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
    deadline = time.time() + 130
    
    # Generate powerful numbers up to 10^9
    limit = 10**9
    powerful = _generate_powerful(limit)
    members = set(powerful)
    
    # Filter for small powerful numbers to compute pairwise sums
    # We assume the optimal N is formed by relatively small powerful numbers
    # due to the decreasing density of powerful numbers.
    small_limit = 10**8
    small_powerful = [p for p in powerful if p <= small_limit]
    
    if len(small_powerful) < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    arr = np.array(small_powerful, dtype=np.int64)
    n_small = len(arr)
    
    counts = {}
    
    # Compute pairwise sums of small_powerful
    # Chunked to avoid memory issues
    chunk_size = 500
    
    for start_i in range(0, n_small, chunk_size):
        end_i = min(start_i + chunk_size, n_small)
        
        # For each i in [start_i, end_i), compute sums with j >= i
        # We can vectorize this by taking arr[i:] for each i?
        # No, arr[i:] changes size.
        # Instead, for a chunk of i's, we can compute sums with ALL j >= start_i?
        # But we need j >= i.
        # If we compute sums for i in chunk with j in [start_i, n_small),
        # we will include pairs where j < i (if j is in the same chunk and j < i).
        # We can filter these out later or just allow them and divide?
        # No, distinct pairs a <= b.
        
        # Simpler: for each i in the chunk, compute arr[i] + arr[i:]
        # This is O(chunk_size * n_small) operations.
        
        for i in range(start_i, end_i):
            a = arr[i]
            sums = a + arr[i:]
            
            # Count sums
            # Using np.unique is O(N log N).
            # For 10k elements, it's fast.
            unique_sums, cnts = np.unique(sums, return_counts=True)
            
            for s, c in zip(unique_sums, cnts):
                s_int = int(s)
                counts[s_int] = counts.get(s_int, 0) + int(c)
        
        if time.time() > deadline - 30:
            break
    
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find top 5 candidates
    top_candidates = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    best_N = 0
    best_count = 0
    best_pairs = []
    
    for N, _ in top_candidates:
        # Verify N against the FULL list of powerful numbers up to N
        # This captures any representations involving powerful numbers > small_limit
        # (though if N < 2*small_limit, a and b must be < N, so they are in small_powerful
        #  unless N > 2*small_limit)
        
        pairs = []
        # Iterate through powerful numbers a <= N/2
        # We can use the full 'powerful' list
        for a in powerful:
            if 2 * a > N:
                break
            b = N - a
            if b in members:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
            
        if time.time() > deadline - 10:
            break
            
    if best_N == 0:
        # Fallback
        best_N = 2
        best_pairs = [[1, 1]]
        
    return {'N': best_N, 'pairs': best_pairs}
