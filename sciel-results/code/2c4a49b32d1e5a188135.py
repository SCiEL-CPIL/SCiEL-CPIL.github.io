import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)

    def sieve_primes(limit):
        """Sieve of Eratosthenes to find all primes up to limit."""
        if limit < 2:
            return np.array([], dtype=np.int64)
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.nonzero(is_prime)[0]

    def compute_phi_numpy(lo, hi, primes):
        """
        Compute phi(n) for all n in [lo, hi] using NumPy.
        """
        n = hi - lo + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        
        vals = np.arange(lo, hi + 1, dtype=np.int64)
        phi = vals.copy()
        
        # Handle phi(1) if lo == 1
        if lo == 1:
            phi[0] = 1
        
        # For each prime p, update phi for multiples of p
        for p in primes:
            if p > hi:
                break
            # Find start index of first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
            
            # Indices in the array corresponding to multiples of p
            # start, start+p, start+2p, ...
            idxs = np.arange(start, hi + 1, p) - lo
            
            # Update phi: phi(x) = phi(x) * (1 - 1/p)
            # phi[idxs] = phi[idxs] - phi[idxs] // p
            # Vectorized update
            phi[idxs] = phi[idxs] - (phi[idxs] // p)
            
        return phi

    def find_best_run(phi_arr, offset, max_len=200):
        """
        Find the longest run of distinct phi values in phi_arr.
        Returns (start_index_in_original, length).
        """
        n = len(phi_arr)
        best_k = 0
        best_s = offset
        
        # Sliding window
        # Since we just need the longest run, and k is small (<= 200),
        # we can use a set for each window start, but that's O(N*K).
        # Better: iterate and keep a set, break on collision.
        
        for i in range(n):
            if n - i < best_k + 1:
                break
            seen = set()
            k = 0
            while i + k < n and k < max_len:
                val = phi_arr[i + k]
                if val in seen:
                    break
                seen.add(val)
                k += 1
            if k > best_k:
                best_k = k
                best_s = offset + i
        
        return best_s, best_k

    # Precompute primes up to 10^6 (sqrt(10^12))
    max_prime_limit = 10**6
    primes = sieve_primes(max_prime_limit)
    
    best_s = 1
    best_k = 1
    
    # Search strategy:
    # Focus on large numbers near 10^12.
    # Segments of size 50,000.
    # Move downwards from 10^12.
    
    MAX_N = 10**12
    SEG_SIZE = 50000
    
    # Start from a high value and move down
    # We can't search the entire range, so we sample.
    # Let's try a few specific high ranges and then random high ranges.
    
    # 1. Deterministic high ranges
    test_starts = [
        MAX_N - 1000000,
        MAX_N - 5000000,
        MAX_N - 10000000,
        MAX_N - 50000000,
        MAX_N - 100000000,
        MAX_N - 500000000,
        MAX_N - 1000000000,
    ]
    
    for start in test_starts:
        if time_left() < 5.0:
            break
        lo = max(1, start)
        hi = min(MAX_N, lo + SEG_SIZE - 1)
        try:
            phi_arr = compute_phi_numpy(lo, hi, primes)
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue

    # 2. Random search in high range
    # Use a simple random generator for reproducibility within the run
    import random
    random.seed(42)
    
    attempts = 0
    while time_left() > 10.0 and attempts < 200:
        attempts += 1
        # Sample lo uniformly in [10^11, 10^12]
        lo = random.randint(10**11, MAX_N - SEG_SIZE)
        hi = lo + SEG_SIZE - 1
        
        try:
            phi_arr = compute_phi_numpy(lo, hi, primes)
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue

    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > MAX_N:
        best_k = min(best_k, MAX_N - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
