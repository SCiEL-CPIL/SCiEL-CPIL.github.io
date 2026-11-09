import time
import bisect
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful_set = set([1])
    sqrt_limit = isqrt(limit)
    
    # Generate using n = m^2 * b where b is squarefree and b | m
    for m in range(1, sqrt_limit + 1):
        temp = m
        factors = []
        for p in range(2, isqrt(temp) + 1):
            while temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)
        
        unique_primes = list(set(factors))
        n_primes = len(unique_primes)
        
        # Enumerate squarefree divisors of m
        # b must be a squarefree divisor of m
        # And m^2 * b <= limit
        # We can just iterate all subsets
        # Optimization: if m^2 > limit, break? No, b>=1, so m^2 <= limit.
        # sqrt_limit ensures m^2 <= limit.
        
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= unique_primes[i]
            
            if b > limit // (m * m):
                continue
            powerful_set.add(m * m * b)
            
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    L = 100_000_000  # 10^8
    
    powerful = _generate_powerful(L)
    powerful_set = set(powerful)
    powerful_np = np.array(powerful, dtype=np.int32)
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback to smaller search if generation was slow
        L = 10_000_000
        powerful = _generate_powerful(L)
        powerful_set = set(powerful)
        powerful_np = np.array(powerful, dtype=np.int32)

    max_N = 2 * L
    # counts array
    # Size max_N + 1
    # Use int32 to save memory
    try:
        counts = np.zeros(max_N + 1, dtype=np.int32)
    except MemoryError:
        # Reduce L if memory error
        L = 20_000_000
        powerful = _generate_powerful(L)
        powerful_set = set(powerful)
        powerful_np = np.array(powerful, dtype=np.int32)
        max_N = 2 * L
        counts = np.zeros(max_N + 1, dtype=np.int32)

    P = len(powerful_np)
    powerful_list = list(powerful_np)
    
    for i in range(P):
        a = powerful_np[i]
        if 2 * a > max_N:
            break
            
        # b >= a, b <= max_N - a
        lo = bisect.bisect_left(powerful_list, a)
        hi = bisect.bisect_right(powerful_list, max_N - a)
        
        if lo < hi:
            b_vals = powerful_np[lo:hi]
            sums = a + b_vals
            counts[sums] += 1
            
        if time.time() > deadline - 5:
            break

    # Find max
    # counts[0] and counts[1] are 0
    best_N = np.argmax(counts)
    best_count = counts[best_N]
    
    # Reconstruct pairs
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': int(best_N), 'pairs': pairs}
