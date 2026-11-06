"""Smooth two-summand decomposition of 123457."""
import time
import heapq

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457

    # Step 1: Compute largest prime factor P(n) for all n up to TARGET using sieve
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i

    best_val = float('inf')
    best_pair = (1, TARGET - 1)

    # Step 2: Generate smooth numbers (with small prime factors) up to TARGET
    # Use primes up to 100 for generating candidates
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Generate all numbers <= TARGET with only these prime factors
    smooth_numbers = set()
    heap = [1]
    smooth_numbers.add(1)
    
    while heap:
        x = heapq.heappop(heap)
        if x > TARGET:
            continue
        for p in small_primes:
            nx = x * p
            if nx > TARGET:
                break
            if nx not in smooth_numbers:
                smooth_numbers.add(nx)
                heapq.heappush(heap, nx)
    
    smooth_list = sorted(smooth_numbers)
    
    # Step 3: For each smooth a, check if b = TARGET - a is also "smooth enough"
    # We want max(P(a), P(b)) to be small
    for a in smooth_list:
        if time.time() > deadline - 5.0:
            break
        b = TARGET - a
        if b < 1:
            continue
        pa = lpf[a]
        pb = lpf[b]
        val = max(pa, pb)
        if val < best_val:
            best_val = val
            best_pair = (a, b)
            # If we've achieved a very good value, we can potentially stop early
            # but let's continue to be safe
            if best_val <= 50:
                # Very good - both have only small prime factors
                # Continue to see if we can do even better, but this is likely near-optimal
                pass
    
    # Step 4: If time remains, do a broader search over all a
    # But focus on ranges where we might find better solutions
    if time.time() < deadline - 10.0:
        # Check a range around the best found, and also check small a values
        # Since the search is symmetric (a,b) and (b,a), we only need a <= TARGET//2
        # but let's check all to be safe, with early termination
        for a in range(1, TARGET // 2 + 1):
            if time.time() > deadline - 5.0:
                break
            b = TARGET - a
            pa = lpf[a]
            pb = lpf[b]
            val = max(pa, pb)
            if val < best_val:
                best_val = val
                best_pair = (a, b)
                if best_val <= 30:
                    # Excellent result
                    break
    
    a, b = best_pair
    if time.time() > deadline:
        return [1, TARGET - 1]
    return [a, b]
