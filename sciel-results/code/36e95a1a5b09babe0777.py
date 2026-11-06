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
    deadline = time.time() + 130  # 20s safety margin before 150s kill
    
    # Use a larger limit to get more powerful numbers
    # With limit ~2e7, we should get ~15-20k powerful numbers
    limit = 20_000_000
    powerful = _generate_powerful(limit)
    members = set(powerful)
    n_pow = len(powerful)
    
    import numpy as np
    arr = np.array(powerful, dtype=np.int64)
    
    # Use np.unique with return_counts on pairwise sums for efficient counting
    # Process in chunks to manage memory
    best_N = 0
    best_count = 0
    
    # We need to count all pairs (a, b) with a <= b such that a + b = N
    # Using vectorized approach: for each chunk of a values, compute sums with all b >= a
    
    # Strategy: use the full array approach with chunked processing
    # For each i, compute arr[i:] + arr[i] and count
    
    # Better: use the fact that we can process in large chunks
    # The key insight from the knowledge: np.unique with return_counts on pairwise sums
    
    # Let's use a smarter approach: process in chunks of i values
    chunk_size = 500
    counts = {}
    
    for start_i in range(0, n_pow, chunk_size):
        end_i = min(start_i + chunk_size, n_pow)
        
        # For each i in [start_i, end_i), we need sums of arr[i] + arr[j] for j >= i
        # This is tricky because j >= i constraint varies per i
        
        # Alternative: just do the full O(n^2) but vectorized per row
        # For each i, compute arr[i] + arr[i:] and accumulate counts
        
        # Use np.bincount for each row - but sums can be large
        # Max sum is 2*limit = 4e7, so bincount array would be 4e7+1 ~ 320MB - too much
        
        # Better approach: use np.unique with return_counts on the concatenated sums
        # But that's still O(n^2) memory
        
        # Let's try a different approach: for each possible N in a range, count how many pairs sum to N
        # But N can be up to 2*limit = 4e7, and for each N we'd need to check all pairs
        
        # Most practical: iterate i, use numpy for the inner loop
        for i in range(start_i, end_i):
            a = arr[i]
            # b must be >= a, so j >= i
            # sums = a + arr[i:]
            sums = a + arr[i:]
            
            # Count occurrences of each sum
            # Use np.unique with return_counts
            unique_sums, cnts = np.unique(sums, return_counts=True)
            
            for s, c in zip(unique_sums, cnts):
                s_int = int(s)
                counts[s_int] = counts.get(s_int, 0) + int(c)
        
        if time.time() > deadline - 15:
            break
    
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Collect all pairs for best_N
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in members and a <= b:
            pairs.append([a, b])
    
    pairs = sorted(set(tuple(p) for p in pairs))
    pairs = [list(p) for p in pairs]
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
