"""Improved program: segmented sieve for long runs of distinct totients."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute primes up to sqrt(10^12) = 10^6 for segmented sieve
    def sieve_primes(limit):
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes_1m = sieve_primes(10**6)

    def segmented_phi(start, length):
        """
        Compute phi(n) for n in [start, start+length-1] using segmented sieve.
        Returns list of phi values.
        """
        # Initialize with n itself (phi(n) starts as n)
        phi_vals = [start + i for i in range(length)]
        remaining = list(range(length))  # indices still needing processing
        
        for p in primes_1m:
            if p * p > start + length - 1 and all(phi_vals[i] == 1 or start+i <= 1 for i in range(length)):
                break
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            if first < start:
                first = start
            for m in range(first, start + length, p):
                idx = m - start
                if phi_vals[idx] > 1:
                    phi_vals[idx] -= phi_vals[idx] // p
                    # Mark that we've processed this prime for this number
                    # We don't need to track remaining separately for correctness,
                    # but we can skip if already reduced to 1
        # Handle any remaining prime factors > sqrt(max_n)
        for i in range(length):
            if phi_vals[i] > 1:
                n = start + i
                # If n > 1 and not fully factored by small primes, 
                # the remaining factor is prime
                # Actually, if n had a prime factor > 10^6, it would be at most one such factor
                # (since (10^6)^2 = 10^12)
                # So if phi_vals[i] > 1 after dividing by all small primes,
                # the remaining cofactor is prime
                phi_vals[i] -= phi_vals[i] // phi_vals[i]
        
        return phi_vals

    def find_longest_distinct_run(phi_vals, offset=0):
        """Find the longest run of consecutive distinct values in phi_vals."""
        best_k = 0
        best_start = offset
        seen = set()
        run_start = 0
        run_len = 0

        for i, v in enumerate(phi_vals):
            if v not in seen:
                seen.add(v)
                run_len += 1
            else:
                if run_len > best_k:
                    best_k = run_len
                    best_start = offset + run_start
                seen = {v}
                run_start = i
                run_len = 1

        if run_len > best_k:
            best_k = run_len
            best_start = offset + run_start

        return best_k, best_start

    best_k = 1
    best_start = 1

    # Phase 1: Use segmented sieve on large blocks
    # Start with a large block near 1 to find initial good runs
    block_size = 10**7  # 10 million numbers per block
    
    # Try several blocks at different locations
    # Start with block starting at 1
    if time_remaining() > 10.0:
        try:
            phi_block = segmented_phi(1, block_size)
            k, s = find_longest_distinct_run(phi_block, offset=1)
            if k > best_k:
                best_k = k
                best_start = s
        except (MemoryError, Exception) as e:
            pass

    # Try more blocks at different locations
    # Search in regions that might have longer runs
    # Based on number theory, runs of distinct phi values are more common
    # in ranges with varied factorization patterns
    
    search_starts = [
        10**6, 10**7, 10**8, 10**9, 10**10, 10**11,
        2*10**6, 5*10**6, 2*10**7, 5*10**7,
        2*10**8, 5*10**8, 2*10**9, 5*10**9,
        2*10**10, 5*10**10, 2*10**11, 5*10**11,
    ]
    
    # Add some random starts
    random.seed(42)
    for _ in range(20):
        s = random.randint(10**6, 10**12 - block_size)
        search_starts.append(s)
    
    # Sort to process in order and avoid duplicates
    search_starts = sorted(set(search_starts))
    
    for s in search_starts:
        if time_remaining() < 5.0:
            break
        if s < 1 or s + block_size - 1 > 10**12:
            continue
        try:
            phi_block = segmented_phi(s, block_size)
            k, start = find_longest_distinct_run(phi_block, offset=s)
            if k > best_k:
                best_k = k
                best_start = start
        except (MemoryError, Exception) as e:
            pass

    # Phase 2: Local refinement around best found
    if best_k > 1 and time_remaining() > 3.0:
        # Try smaller blocks around the best start to refine
        refine_size = 100000
        for delta in range(-refine_size, refine_size + 1, 1000):
            if time_remaining() < 1.0:
                break
            s = best_start + delta
            if s < 1 or s + refine_size - 1 > 10**12:
                continue
            try:
                phi_block = segmented_phi(s, refine_size)
                k, start = find_longest_distinct_run(phi_block, offset=s)
                if k > best_k:
                    best_k = k
                    best_start = start
            except (MemoryError, Exception) as e:
                pass

    # Ensure constraints
    if best_k > 200:
        best_k = 200
    if best_start + best_k - 1 > 10**12:
        best_start = 10**12 - best_k + 1
    if best_start < 1:
        best_start = 1
    if best_k < 1:
        best_k = 1

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
