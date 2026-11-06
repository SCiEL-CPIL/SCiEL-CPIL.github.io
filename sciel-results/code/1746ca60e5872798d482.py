"""Improved program: segmented sieve for long runs of distinct totients."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # Precompute small primes for trial division
    def sieve_primes(limit):
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = 0
        return [i for i in range(2, limit + 1) if is_prime[i]]

    small_primes = sieve_primes(100000)  # primes up to 100000 for quick factorization

    def phi_trial(x, primes=None):
        """Compute phi(x) using trial division by primes."""
        if primes is None:
            primes = small_primes
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    def find_longest_distinct_run(phi_vals, offset=0):
        """Find the longest run of consecutive distinct values in phi_vals.
        Returns (best_k, best_start) where best_start is the actual number (offset + index)."""
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

    def check_window(start, length, primes=None):
        """Check if a window has all distinct phi values, return max distinct prefix length."""
        if primes is None:
            primes = small_primes
        seen = set()
        k = 0
        for i in range(length):
            v = phi_trial(start + i, primes)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    # Segmented sieve for computing phi over a range [low, high]
    def segmented_phi(low, high, primes=None):
        """Compute phi(n) for all n in [low, high] using segmented sieve.
        Returns a list of phi values indexed by (n - low)."""
        if primes is None:
            primes = small_primes
        
        n = high - low + 1
        # Initialize phi array: phi[i] = low + i initially, then multiply by (1 - 1/p) for each prime factor
        phi = [low + i for i in range(n)]
        
        for p in primes:
            if p * p > high:
                break
            # Find the first multiple of p in [low, high]
            first = ((low + p - 1) // p) * p
            if first < p:
                first = p
            for j in range(first, high + 1, p):
                idx = j - low
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        
        # Handle remaining prime factors > sqrt(high)
        for i in range(n):
            val = low + i
            if phi[i] != val:
                # Already processed some factors
                pass
            # Check if there's a remaining prime factor
            y = val
            for p in primes:
                if p * p > y:
                    break
                while y % p == 0:
                    y //= p
            if y > 1:
                phi[i] -= phi[i] // y
        
        return phi

    best_k = 1
    best_start = 1

    # Phase 1: Segmented sieve search over large ranges
    # Search in chunks of size CHUNK_SIZE
    CHUNK_SIZE = 5 * 10**6
    
    # Search from 1 up to as far as time permits
    search_start = 1
    while time_remaining() > 10.0:
        chunk_end = min(search_start + CHUNK_SIZE - 1, 10**12)
        try:
            phi_vals = segmented_phi(search_start, chunk_end)
            k, s = find_longest_distinct_run(phi_vals, search_start)
            if k > best_k:
                best_k = k
                best_start = s
                if best_k >= 200:
                    break
        except MemoryError:
            # Try smaller chunk
            CHUNK_SIZE = 10**6
            continue
        
        search_start = chunk_end + 1
        if search_start > 10**12:
            break

    # Phase 2: Targeted search in larger ranges using trial division
    # Regions where numbers have varied factorizations tend to have distinct phi values
    
    random.seed(42)
    
    # Try windows around multiples of primorials
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870]
    
    for prim in primorials:
        if time_remaining() < 5.0:
            break
        # Try a few multiples
        max_mult = min(10**12 // prim, 10000)
        for _ in range(min(50, max_mult)):
            if time_remaining() < 3.0:
                break
            mult = random.randint(1, max_mult)
            start = mult * prim
            if start < 1 or start + 200 > 10**12:
                continue
            k = check_window(start, 200)
            if k > best_k:
                best_k = k
                best_start = start
            # Also try start - 1, start + 1, etc.
            for offset in [-10, -5, -1, 1, 5, 10]:
                s2 = start + offset
                if s2 < 1 or s2 + 200 > 10**12:
                    continue
                k2 = check_window(s2, 200)
                if k2 > best_k:
                    best_k = k2
                    best_start = s2

    # Phase 3: Random search in mid-range
    if time_remaining() > 5.0 and best_k < 200:
        for _ in range(200):
            if time_remaining() < 2.0:
                break
            s = random.randint(10**6, 10**11)
            if s + 200 > 10**12:
                s = 10**12 - 200
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

    # Phase 4: Local refinement around best found
    if best_k > 1 and time_remaining() > 3.0:
        # Try expanding around the best start
        for delta in range(-50, 51):
            if time_remaining() < 1.0:
                break
            s = best_start + delta
            if s < 1 or s + 200 > 10**12:
                continue
            k = check_window(s, 200)
            if k > best_k:
                best_k = k
                best_start = s

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
