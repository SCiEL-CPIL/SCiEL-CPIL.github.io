import numpy as np
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def sieve_primes(n):
        """Sieve of Eratosthenes up to n."""
        if n < 2:
            return np.array([], dtype=np.int64)
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]
    
    def phi_sieve(n):
        """Compute phi(1..n) using sieve method."""
        phi = np.arange(n + 1, dtype=np.int64)
        for i in range(2, n + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, n + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    def factorize(x, primes):
        """Factorize x using precomputed primes. Returns list of (p, e)."""
        factors = []
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                e = 0
                while x % p == 0:
                    x //= p
                    e += 1
                factors.append((p, e))
        if x > 1:
            factors.append((x, 1))
        return factors
    
    # Precompute primes up to sqrt(10^12) = 10^6
    primes = sieve_primes(10**6)
    
    def phi_exact(x):
        """Exact Euler's totient function."""
        if x == 1:
            return 1
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
    
    def find_best_run(start, length, phi_cache, primes):
        """Check if phi values in [start, start+length-1] are pairwise distinct."""
        seen = set()
        for i in range(length):
            x = start + i
            if x in phi_cache:
                v = phi_cache[x]
            else:
                v = phi_exact(x)
                phi_cache[x] = v
            if v in seen:
                return False
            seen.add(v)
        return True
    
    def search_region(lo, hi, max_k, phi_cache, primes):
        """Search for the longest run of distinct phi values in [lo, hi]."""
        best_k = 0
        best_s = lo
        
        # First, compute phi for the entire region
        vals = {}
        for x in range(lo, hi + 1):
            if x in phi_cache:
                vals[x] = phi_cache[x]
            else:
                vals[x] = phi_exact(x)
                phi_cache[x] = vals[x]
        
        # Slide window to find longest run of distinct values
        for s in range(lo, hi - max_k + 2):
            # Check if we can extend from previous best
            k = 1
            seen = {vals[s]}
            while s + k <= hi and k < max_k:
                v = vals[s + k]
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= max_k:
                    break
        
        return best_s, best_k
    
    phi_cache = {}
    
    # Strategy: Search in regions, starting from small numbers and going up
    # The known best runs are relatively short, so we search adaptively
    
    # Start with a moderate region to get a baseline
    initial_regions = [
        (1, 100000),
        (100001, 200000),
        (200001, 500000),
    ]
    
    best_k_overall = 1
    best_s_overall = 1
    
    for lo, hi in initial_regions:
        if time.time() > deadline - 10:
            break
        s, k = search_region(lo, hi, 200, phi_cache, primes)
        if k > best_k_overall:
            best_k_overall = k
            best_s_overall = s
    
    # Now search more broadly, focusing on areas where collisions are less likely
    # Numbers with many distinct prime factors tend to have larger phi values
    # which are less likely to collide
    
    # Search in larger steps, checking windows of size 200
    step = 10000
    max_search = 10**12
    
    # Adaptive search: try different starting points
    # Focus on numbers that are products of distinct primes (squarefree numbers)
    # as they tend to have larger phi values
    
    # First pass: scan in chunks
    chunk_size = 500000
    current = 1
    while current < max_search and time.time() < deadline - 15:
        hi = min(current + chunk_size - 1, max_search)
        
        # Sample: check a few windows in this chunk
        # Start with the beginning of the chunk
        s, k = search_region(current, min(current + 100000, hi), 200, phi_cache, primes)
        if k > best_k_overall:
            best_k_overall = k
            best_s_overall = s
        
        # Also check the middle
        mid = current + (hi - current) // 2
        if mid + 100000 <= hi:
            s, k = search_region(mid, mid + 100000, 200, phi_cache, primes)
            if k > best_k_overall:
                best_k_overall = k
                best_s_overall = s
        
        current = hi + 1
        
        # If we found a good run, search more carefully around it
        if best_k_overall > 10:
            # Expand search around the best found
            expand_lo = max(1, best_s_overall - 100000)
            expand_hi = min(max_search, best_s_overall + 100000)
            s, k = search_region(expand_lo, expand_hi, 200, phi_cache, primes)
            if k > best_k_overall:
                best_k_overall = k
                best_s_overall = s
    
    # Final polish: if we have time, do a more careful search around the best
    if time.time() < deadline - 5 and best_k_overall > 1:
        # Search a wider window around the best start
        wide_lo = max(1, best_s_overall - 500000)
        wide_hi = min(max_search, best_s_overall + 500000)
        s, k = search_region(wide_lo, wide_hi, 200, phi_cache, primes)
        if k > best_k_overall:
            best_k_overall = k
            best_s_overall = s
    
    # Verify the result
    if best_s_overall + best_k_overall - 1 > 10**12:
        best_k_overall = max(1, 10**12 - best_s_overall + 1)
    
    return {'start': best_s_overall, 'length': best_k_overall}
