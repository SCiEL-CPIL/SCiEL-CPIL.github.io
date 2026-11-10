"""Optimized program for few distinct small-prime parts in short integer interval."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(small_primes)

    # Max value we need: n+24 where n <= 1000000
    max_val = 1000024

    # Precompute sf[x] = product of prime factors p < 24 (with multiplicity)
    sf = np.ones(max_val + 1, dtype=np.int64)

    for p in small_primes:
        pk = p
        while pk <= max_val:
            # Multiply all multiples of pk by p
            sf[pk::pk] *= p
            pk *= p
        if time.time() > deadline - 30:
            break

    # sf[x] is the complete factor composed of primes strictly below 24
    # For x with no small prime factors, sf[x] = 1

    min_n = 1000
    max_n = 1000000
    window_size = 24

    # Strategy: Use a sliding window approach with vectorized distinct counting
    # Process in chunks to stay within time budget

    best_score = 25
    best_n = min_n

    # First, do a quick scan to find a good initial candidate
    # Check windows at multiples of 2520 (lcm of 1..10) and nearby
    # These are more likely to have many even numbers, multiples of 3, etc.

    # Quick heuristic scan: check n values that are multiples of small numbers
    # to increase the chance of shared factors
    step = 120  # check every 120th n initially for speed
    checked = 0
    for n in range(min_n, max_n + 1, step):
        if time.time() > deadline - 20:
            break
        window = sf[n+1:n+25]
        # Count distinct values
        s = np.sort(window)
        count = 1 + np.sum(s[1:] != s[:-1])
        if count < best_score:
            best_score = int(count)
            best_n = n
            if best_score <= 2:
                break
        checked += 1

    # Now do a denser search around the best candidate found
    if best_n > min_n and best_n < max_n:
        # Search in a window around the best candidate
        search_radius = 5000
        lo = max(min_n, best_n - search_radius)
        hi = min(max_n, best_n + search_radius)
        
        for n in range(lo, hi + 1):
            if time.time() > deadline - 10:
                break
            window = sf[n+1:n+25]
            s = np.sort(window)
            count = 1 + np.sum(s[1:] != s[:-1])
            if count < best_score:
                best_score = int(count)
                best_n = n
                if best_score <= 2:
                    break

    # If we haven't found a very good score, do a broader vectorized scan
    if best_score > 3:
        chunk_size = 100000
        for chunk_start in range(min_n, max_n + 1, chunk_size):
            if time.time() > deadline - 5:
                break
            chunk_end = min(chunk_start + chunk_size, max_n + 1)
            chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)
            
            # Build 2D array of sf values for all windows in chunk
            indices = chunk_n[:, None] + np.arange(1, 25)[None, :]
            window_2d = sf[indices]
            
            # Sort each row and count distincts
            sorted_rows = np.sort(window_2d, axis=1)
            diffs = np.diff(sorted_rows, axis=1)
            distinct_counts = 1 + np.sum(diffs != 0, axis=1)
            
            min_count = int(distinct_counts.min())
            if min_count < best_score:
                best_score = min_count
                best_n = int(chunk_n[distinct_counts.argmin()])
                if best_score <= 2:
                    break

    # Final polish: if best_score is still > 2, do another targeted search
    if best_score > 2 and time.time() < deadline - 3:
        # Try specific promising n values: near highly composite numbers
        # and check a dense neighborhood
        candidates = []
        for base in range(1000, 1000001, 2520):
            candidates.append(base)
            candidates.append(base - 1)
            candidates.append(base + 1)
        
        # Also add the current best_n neighborhood
        candidates.extend(range(max(min_n, best_n - 200), min(max_n, best_n + 200) + 1))
        
        # Deduplicate and filter to valid range
        candidates = sorted(set(c for c in candidates if min_n <= c <= max_n))
        
        for n in candidates:
            if time.time() > deadline - 2:
                break
            window = sf[n+1:n+25]
            s = np.sort(window)
            count = 1 + np.sum(s[1:] != s[:-1])
            if count < best_score:
                best_score = int(count)
                best_n = n
                if best_score <= 2:
                    break

    return best_n
# EVOLVE-BLOCK-END
