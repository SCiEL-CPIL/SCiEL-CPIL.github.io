# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt
from collections import defaultdict
import bisect

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful_set = set()
    powerful_set.add(1)
    
    sqrt_limit = isqrt(limit)
    
    for m in range(1, sqrt_limit + 1):
        # Factor m to get its distinct prime factors
        temp = m
        primes = []
        for p in range(2, isqrt(temp) + 1):
            if temp % p == 0:
                primes.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            primes.append(temp)
        
        # Enumerate all subsets of primes to get squarefree divisors
        n_primes = len(primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            if b > limit // (m * m):
                continue
            powerful_set.add(m * m * b)
    
    return np.array(sorted(powerful_set), dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Increase N_MAX significantly due to NumPy speedup
    # 2 * 10^8 allows for ~15,000 powerful numbers.
    # Counting array size 2*10^8 * 2 bytes (int16) = 400MB.
    # If memory is tight, reduce to 10^8.
    N_MAX = 200_000_000
    
    # Check if we have enough time and memory
    # Generating powerful numbers up to 2*10^8 takes a few seconds.
    powerful = _generate_powerful(N_MAX)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    P = len(powerful)
    
    # Use int16 for counts. Max representations for N ~ 10^8 is likely < 100.
    # If overflow occurs, it wraps, but we only need argmax.
    # To be safe against overflow if counts are high, we could use int32, 
    # but that doubles memory to 800MB. Let's try int16 first.
    # Actually, let's use int32 to be safe, and reduce N_MAX to 100_000_000 if needed.
    # Let's try N_MAX = 100_000_000 with int32 (400MB) or int16 (200MB).
    # Let's stick to N_MAX = 100_000_000 to be safe on memory, but use int16.
    # Wait, the previous program did 50M in Python. NumPy is ~100x faster for the inner loop.
    # So we can do 50M * 100 = 5 * 10^9? No, memory is the limit.
    # Memory limit is the bottleneck.
    # Let's use N_MAX = 100_000_000.
    # P ~ 1.1 * 10^4 = 11000.
    # Ops ~ 11000 * 5500 = 6 * 10^7. Very fast.
    
    # Let's try N_MAX = 200_000_000 with int16.
    # If MemoryError, fall back to smaller.
    
    try:
        count_array = np.zeros(N_MAX + 1, dtype=np.int16)
    except MemoryError:
        N_MAX = 50_000_000
        powerful = _generate_powerful(N_MAX)
        count_array = np.zeros(N_MAX + 1, dtype=np.int16)
        P = len(powerful)

    # Count representations
    for i in range(P):
        a = powerful[i]
        if 2 * a > N_MAX:
            break
        
        if time.time() > deadline - 20:
            break
            
        # Find b in powerful such that a <= b <= N_MAX - a
        # powerful is sorted
        lo = bisect.bisect_left(powerful, a)
        hi = bisect.bisect_right(powerful, N_MAX - a)
        
        if lo >= hi:
            continue
            
        b_vals = powerful[lo:hi]
        sums = a + b_vals
        
        # np.add.at is slower than bincount but works with sparse indices?
        # Actually, sums are dense enough? No, they are specific values.
        # np.add.at is the correct way to accumulate into an array at specific indices.
        np.add.at(count_array, sums, 1)
        
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # If we ran out of time, return best so far
        if count_array.any():
            best_N = np.argmax(count_array)
            pairs = _reconstruct_pairs(best_N, powerful, N_MAX)
            return {'N': int(best_N), 'pairs': pairs}
        else:
            return {'N': 2, 'pairs': [[1, 1]]}

    best_N = np.argmax(count_array)
    pairs = _reconstruct_pairs(best_N, powerful, N_MAX)
    
    return {'N': int(best_N), 'pairs': pairs}

def _reconstruct_pairs(N, powerful, N_MAX):
    """Reconstruct pairs for a given N."""
    powerful_set = set(powerful) # Convert to set for O(1) lookup
    pairs = []
    # Iterate a from 1 to N//2
    # We can iterate over powerful numbers up to N//2
    for a in powerful:
        if a > N // 2:
            break
        b = N - a
        if b >= a and b in powerful_set:
            pairs.append([int(a), int(b)])
    return pairs
# EVOLVE-BLOCK-END
