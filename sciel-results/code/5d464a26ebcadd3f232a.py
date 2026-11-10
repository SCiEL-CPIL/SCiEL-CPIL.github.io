import sys
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 60.0
    deadline = start_time + timeout - 3.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Vectorized sieve: for each prime p and each power p^k,
    # multiply small_part at indices divisible by p^k by p.
    small_part = np.ones(size + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= size:
            small_part[power::power] *= p
            power *= p
    
    sp = small_part
    
    # Helper: compute distinct count for window starting at n
    # Window covers n+1 to n+24
    def distinct_count(n):
        window = sp[n+1:n+25]
        return len(set(window.tolist()))
    
    best_n = 1000
    best_count = 25  # Max possible distinct is 24
    
    # Phase 1: Random sampling to find promising regions
    # Sample ~2000 random positions to get a sense of the landscape
    random.seed(42)
    sample_size = min(2000, max_n - 1000 + 1)
    samples = set()
    for _ in range(sample_size):
        n = random.randint(1000, max_n)
        samples.add(n)
    
    # Evaluate all samples
    sample_results = []
    for n in samples:
        dc = distinct_count(n)
        sample_results.append((dc, n))
    
    sample_results.sort()
    
    # Update best from samples
    for dc, n in sample_results:
        if dc < best_count:
            best_count = dc
            best_n = n
    
    if best_count <= 2:
        return best_n
    
    # Phase 2: Exhaustive search in promising regions
    # Take the top 50 best samples and search ±500 around each
    search_regions = set()
    for dc, n in sample_results[:50]:
        lo = max(1000, n - 500)
        hi = min(max_n, n + 500)
        search_regions.add((lo, hi))
    
    # Also add a few random regions in case the top samples cluster
    for _ in range(10):
        n = random.randint(1000, max_n)
        lo = max(1000, n - 200)
        hi = min(max_n, n + 200)
        search_regions.add((lo, hi))
    
    # Sort regions and search them
    for lo, hi in sorted(search_regions):
        if time.time() > deadline:
            break
        for n in range(lo, hi + 1):
            if time.time() > deadline:
                break
            dc = distinct_count(n)
            if dc < best_count:
                best_count = dc
                best_n = n
                if best_count <= 2:
                    return best_n
    
    if best_count <= 2:
        return best_n
    
    # Phase 3: Broader sweep if time remains
    # Do a coarser scan with step 10 to find other promising areas
    step = 10
    n = 1000
    while n <= max_n:
        if time.time() > deadline:
            break
        dc = distinct_count(n)
        if dc < best_count:
            best_count = dc
            best_n = n
            # Found something good, do local exhaustive search
            lo = max(1000, n - 50)
            hi = min(max_n, n + 50)
            for m in range(lo, hi + 1):
                if time.time() > deadline:
                    break
                dc2 = distinct_count(m)
                if dc2 < best_count:
                    best_count = dc2
                    best_n = m
                    if best_count <= 2:
                        return best_n
        n += step
    
    return best_n
# EVOLVE-BLOCK-END
