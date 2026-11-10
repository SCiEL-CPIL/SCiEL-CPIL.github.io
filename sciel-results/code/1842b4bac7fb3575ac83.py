"""
Strategy: For small n (< ~10^6), brute-force phi and search for max distinct runs.
For large n, use the insight that phi(n) is even for n >= 3, so long runs must
avoid many coincidences. The current program found k=7 (score 2.78 means raw value
7 with seed 2.5, or similar). Let's try to find longer runs.

Key insight: phi values repeat when there are "phi-collisions". For random n in
[1, 10^12], the probability that phi(a) = phi(b) for a != b in a window of size k
depends on the distribution. Longer windows at larger n tend to have more distinct
values because phi grows roughly linearly.

Better approach: Use the fact that for n > 2, phi(n) is even. So in any window
of consecutive integers of length >= 3, at most 2 values can have odd phi (only
n=1,2 give odd phi). For larger windows, all phi values are even, so we need
to check for collisions among even values.

The key observation: if we can find a window where all phi values are distinct,
that's our answer. Let's do a more careful search.

Actually, let's think about this differently. The maximum k we can achieve is bounded
by the fact that phi(n) takes values in [1, n], and for a window [s, s+k-1], 
the phi values are all in [1, s+k-1]. By pigeonhole, if k > s+k-1, impossible,
but that's not binding. The real constraint is that phi values must be distinct.

For the brute-force approach on small ranges, let's be smarter:
- For n up to ~50000, compute phi efficiently and scan for longest distinct run.
- The answer is likely in the range of a few hundred to a few thousand.

Let's do a systematic search up to n=200000, which should be fast enough.
"""
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_phi_up_to(n):
        """Compute phi(1)..phi(n) using sieve method."""
        if n < 1:
            return np.array([], dtype=np.int64)
        phi = np.arange(1, n + 1, dtype=np.int64)
        for i in range(2, int(math.isqrt(n)) + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi

    def find_longest_distinct_run(phi_arr):
        """Find longest run of distinct values in phi_arr (1-indexed, phi_arr[0]=phi(1))."""
        n = len(phi_arr)
        if n == 0:
            return 1, 1
        best_k = 1
        best_s = 1
        left = 0
        count = {}
        for right in range(n):
            val = int(phi_arr[right])
            count[val] = count.get(val, 0) + 1
            while count[val] > 1:
                left_val = int(phi_arr[left])
                count[left_val] -= 1
                if count[left_val] == 0:
                    del count[left_val]
                left += 1
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = left + 1  # 1-indexed
        return best_s, best_k

    best_s, best_k = 1, 1

    # Phase 1: Search up to n=200000
    if time_left() > 30:
        N1 = 200000
        phi_arr = compute_phi_up_to(N1)
        s, k = find_longest_distinct_run(phi_arr)
        if k > best_k:
            best_k = k
            best_s = s

    # Phase 2: Search in larger ranges using segmented approach
    # For ranges [lo, hi], compute phi and find longest distinct run
    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi]."""
        n = hi - lo + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        
        limit = int(math.isqrt(hi)) + 1
        
        # Sieve primes up to limit
        is_prime = np.ones(limit + 1, dtype=bool)
        if limit >= 2:
            is_prime[0] = False
            is_prime[1] = False
            for i in range(2, int(math.isqrt(limit)) + 1):
                if is_prime[i]:
                    is_prime[i*i::i] = False
            primes = np.where(is_prime)[0]
        
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes:
            first = ((lo + p - 1) // p) * p
            if first > hi:
                continue
            indices = np.arange(first, hi + 1, p) - lo
            mask = (remaining[indices] % p == 0)
            if not np.any(mask):
                continue
            phi[indices[mask]] -= phi[indices[mask]] // p
            for idx in indices[mask]:
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
        
        mask = remaining > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // remaining[mask]
        
        return phi

    # Try various ranges, focusing on where longer runs might exist
    # Based on the observation that larger n tends to give more distinct phi values
    search_ranges = [
        (1000000, 2000000),
        (5000000, 6000000),
        (10000000, 11000000),
        (50000000, 51000000),
        (100000000, 101000000),
        (500000000, 501000000),
        (1000000000, 1001000000),
        (5000000000, 5001000000),
        (10000000000, 10001000000),
        (50000000000, 50001000000),
        (100000000000, 100001000000),
        (500000000000, 500001000000),
        (999990000000, 999991000000),
    ]

    for lo, hi in search_ranges:
        if time_left() < 15:
            break
        if hi > 10**12:
            continue
        try:
            phi_arr = compute_phi_segmented(lo, hi)
            s, k = find_longest_distinct_run(phi_arr)
            if k > best_k:
                best_k = k
                best_s = s + lo - 1  # adjust for offset
        except Exception:
            continue

    # Safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
        best_k = 1

    return {'start': best_s, 'length': best_k}
