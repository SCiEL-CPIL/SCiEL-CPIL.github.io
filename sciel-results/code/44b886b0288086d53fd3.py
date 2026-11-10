"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    _prime_cache = {}

    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray, with caching."""
        if limit in _prime_cache:
            return _prime_cache[limit]
        if limit < 2:
            result = []
        else:
            is_prime = bytearray(b'\x01') * (limit + 1)
            is_prime[0] = is_prime[1] = 0
            for i in range(2, int(math.isqrt(limit)) + 1):
                if is_prime[i]:
                    start = i * i
                    step = i
                    is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
            result = [i for i in range(2, limit + 1) if is_prime[i]]
        _prime_cache[limit] = result
        return result

    def compute_phi_segmented_numpy(lo, hi):
        """Compute phi for numbers in [lo, hi] using NumPy for speed."""
        n = hi - lo + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        limit = int(math.isqrt(hi)) + 1
        primes = sieve_primes(limit)

        # phi[i] represents phi(lo + i), remaining[i] represents the remaining factor
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        remaining = np.arange(lo, hi + 1, dtype=np.int64)

        for p in primes:
            # Find indices in [lo, hi] that are multiples of p
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
            # Vectorized operation for multiples of p
            # Indices: start, start+p, start+2p, ... <= hi
            # idx = j - lo where j is the multiple
            first_idx = start - lo
            # Number of multiples
            count = (hi - start) // p + 1
            idxs = first_idx + np.arange(count) * p

            # Only process if remaining[idx] is divisible by p
            rem_vals = remaining[idxs]
            mask = rem_vals % p == 0
            if np.any(mask):
                valid_idxs = idxs[mask]
                # Update phi: phi -= phi // p
                phi[valid_idxs] -= phi[valid_idxs] // p
                # Update remaining: divide out all factors of p
                r = remaining[valid_idxs].astype(np.int64)
                while np.any(r % p == 0):
                    mask2 = r % p == 0
                    if np.any(mask2):
                        r = r.copy()
                        r[mask2] = r[mask2] // p
                remaining[valid_idxs] = r

        # Handle remaining prime factors
        # For each i, if remaining[i] > 1, it's a prime factor
        mask = remaining > 1
        if np.any(mask):
            idxs = np.where(mask)[0]
            rem = remaining[idxs]
            phi[idxs] -= phi[idxs] // rem

        return phi

    def find_best_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values in phi_arr using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 0
        best_k = 0
        best_s = offset
        left = 0
        seen = {}

        for right in range(n):
            val = int(phi_arr[right])
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = min(current_len, max_len)
                best_s = offset + left
            if best_k >= max_len:
                break

        return best_s, best_k

    best_s, best_k = 1, 1

    # Strategy: At high values (e.g., > 10^11), phi values are very large and sparse,
    # making collisions extremely rare. We aim to find a run of length 200.
    
    # Phase 1: Target high ranges where collisions are negligible.
    # We pick a range of size 200 at a high base.
    # If we find a run of 200, we can return immediately as it's the max allowed.
    
    # Let's try a few high bases.
    # The knowledge suggests 10^11 is a good base.
    # Let's try to verify a run of 200 there.
    
    bases = [
        10**11,
        2 * 10**11,
        5 * 10**11,
        10**12 - 200000,
        10**10,
        10**9,
    ]

    for base in bases:
        if time_left() < 5.0:
            break
        lo = base
        hi = base + 200
        if hi > 10**12:
            hi = 10**12
            lo = hi - 200
        try:
            phi_arr = compute_phi_segmented_numpy(lo, hi)
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    # We found the maximum possible length
                    return {'start': best_s, 'length': best_k}
        except Exception:
            continue

    # If we didn't find 200 in the quick checks, do a broader search
    # to ensure we have a good baseline, though 200 is the goal.
    
    # Search a slightly wider range around the best found so far, or a new promising area.
    # If best_k < 200, we try to find a better one.
    
    if best_k < 200:
        # Try a few more random-looking high bases
        import random
        random.seed(42)
        for _ in range(10):
            if time_left() < 5.0:
                break
            base = random.randint(10**11, 10**12 - 1000)
            lo = base
            hi = base + 200
            if hi > 10**12:
                hi = 10**12
                lo = hi - 200
            try:
                phi_arr = compute_phi_segmented_numpy(lo, hi)
                s, k = find_best_run(phi_arr, lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
                    if best_k >= 200:
                        return {'start': best_s, 'length': best_k}
            except Exception:
                continue

    # Fallback: ensure we have a valid result
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
