import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin
    
    # Generate primes up to 10^6 using sieve
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def phi(x):
        """Compute Euler's totient function using trial division by primes."""
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
    
    def check_run(s, k):
        """Check if phi(s), phi(s+1), ..., phi(s+k-1) are all distinct."""
        seen = set()
        for i in range(k):
            v = phi(s + i)
            if v in seen:
                return False, i
            seen.add(v)
        return True, k
    
    def find_best_in_range(s_start, s_end, max_k=200):
        """Find the best run in a range."""
        best_k = 1
        best_s = s_start
        for s in range(s_start, s_end + 1):
            if time.time() - start_time > DEADLINE:
                break
            # Try to extend from current best
            k = 1
            seen = {phi(s)}
            while s + k <= s_end and s + k - 1 <= 10**12 and k < max_k:
                v = phi(s + k)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
        return best_k, best_s
    
    # Strategy: search in multiple locations
    # Known good regions and random sampling
    best_k = 1
    best_s = 1
    
    # First, do a quick search in small range to establish baseline
    k, s = find_best_in_range(1, 50000, 200)
    if k > best_k:
        best_k, best_s = k, s
    
    # Search in various strategic locations
    # Numbers with many small prime factors tend to have smaller phi values
    # which might create more collisions, so we want regions with varied factors
    
    search_regions = [
        (100000, 200000),
        (500000, 600000),
        (1000000, 1100000),
        (10000000, 10100000),
        (100000000, 100100000),
        (1000000000, 1000100000),
        (10**10, 10**10 + 100000),
        (10**11, 10**11 + 100000),
        (10**12 - 100000, 10**12),
    ]
    
    for lo, hi in search_regions:
        if time.time() - start_time > DEADLINE:
            break
        if lo > 10**12:
            continue
        hi = min(hi, 10**12 - 199)
        if lo >= hi:
            continue
        # Sample within the region
        step = max(1, (hi - lo) // 50)
        for s in range(lo, hi + 1, step):
            if time.time() - start_time > DEADLINE:
                break
            k = 1
            seen = {phi(s)}
            while s + k - 1 <= 10**12 and k < 200:
                v = phi(s + k)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
                # Try to extend a bit more around this point
                for ds in range(-5, 6):
                    s2 = s + ds
                    if s2 < 1 or s2 + 199 > 10**12:
                        continue
                    k2 = 1
                    seen2 = {phi(s2)}
                    while s2 + k2 - 1 <= 10**12 and k2 < 200:
                        v2 = phi(s2 + k2)
                        if v2 in seen2:
                            break
                        seen2.add(v2)
                        k2 += 1
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
    
    # Final verification
    if best_k > 1:
        ok, _ = check_run(best_s, best_k)
        if not ok:
            # Fall back to safe answer
            best_k = 1
            best_s = 1
    
    return {'start': best_s, 'length': best_k}
