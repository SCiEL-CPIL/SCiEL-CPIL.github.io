"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 140.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    # sqrt(10^12) = 10^6, so primes up to 10^6 are sufficient for factorization
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_range(start, length):
        """Compute phi(n) for n in [start, start+length-1] efficiently."""
        if length <= 0:
            return []
        
        # Initialize arrays
        # ph[i] holds the remaining part of (start+i) after dividing out small primes
        # res[i] holds the evolving phi value
        ph = [start + i for i in range(length)]
        res = [start + i for i in range(length)]
        
        # Limit for primes: we only need primes p such that p*p <= max(start+length-1)
        # However, for the sieve logic on a small range, we can just iterate all primes 
        # and check if they have multiples in the range.
        # To be safe and fast, we iterate primes up to sqrt(start+length-1).
        limit = int(math.isqrt(start + length - 1))
        
        for p in primes:
            if p > limit:
                break
            
            # Find the first multiple of p in the range [start, start+length-1]
            first_mult = ((start + p - 1) // p) * p
            if first_mult > start + length - 1:
                continue
                
            for j in range(first_mult, start + length, p):
                idx = j - start
                if ph[idx] % p == 0:
                    res[idx] -= res[idx] // p
                    while ph[idx] % p == 0:
                        ph[idx] //= p
                        
        # Handle remaining prime factors (which must be > limit, so they are prime)
        for i in range(length):
            if ph[i] > 1:
                res[i] -= res[i] // ph[i]
                
        return res

    def find_longest_run(phi_vals, offset, max_k=200):
        """Find the longest run of distinct phi values in phi_vals.
        Returns (start_index_relative, length)."""
        n = len(phi_vals)
        best_len = 0
        best_idx = 0
        i = 0
        while i < n:
            seen = set()
            k = 0
            while i + k < n and k < max_k and phi_vals[i+k] not in seen:
                seen.add(phi_vals[i+k])
                k += 1
            if k > best_len:
                best_len = k
                best_idx = i
            i += 1
        return offset + best_idx, best_len

    best_k = 1
    best_s = 1

    # Strategy 1: Search near the upper limit (10^12) with small windows
    # This is fast and likely to find long runs if they exist there.
    # We scan backwards from 10^12 - 200.
    
    # First, check the immediate vicinity of the upper bound
    base_s = 10**12 - 200
    
    # Check a few specific offsets
    for offset in range(0, 50):
        if time_left() < 10:
            break
        s = base_s - offset
        if s < 1:
            continue
        k = 200
        # We just need to verify if the run is actually 200 or less
        # To save time, we compute phi for 200 numbers and check distinctness
        vals = phi_range(s, 200)
        seen = set()
        cnt = 0
        for v in vals:
            if v in seen:
                break
            seen.add(v)
            cnt += 1
        if cnt > best_k:
            best_k = cnt
            best_s = s
        if best_k == 200:
            break

    # Strategy 2: If we haven't found 200, try larger blocks in different regions.
    # The current program only looked at 100 spots near the top.
    # Let's try some larger blocks to find a "good" region, then refine.
    # However, the task says "Long runs below 10^12 are finite evidence".
    # It's possible that 200 is the max or close to it.
    
    # If we didn't find 200, let's do a broader search in a few random locations
    # or specific locations known to have high density of numbers with many factors?
    # Actually, phi(n) collisions are more likely when numbers share factors.
    # Numbers with many small prime factors might have "smaller" phi values?
    # Or maybe numbers that are primes have phi(n) = n-1, which are distinct for distinct primes.
    # A run of primes would give distinct phis.
    # But we need consecutive integers. Consecutive integers cannot all be prime (except 2,3).
    # So we are looking for a sequence where no two numbers have the same totient.
    
    # Let's try a larger scan near the top if we have time.
    if best_k < 200 and time_left() > 20:
        # Scan a wider range near the top
        # We can check 1000 starts, each checking 200 numbers.
        # That's 200,000 phi computations.
        # With the optimized phi_range, this should be fast.
        
        # Let's scan backwards from base_s
        scan_start = base_s
        scan_end = base_s - 1000
        for s in range(scan_start, scan_end, -1):
            if time_left() < 5:
                break
            if s < 1:
                break
            # Quick check: compute phi for 200 numbers
            vals = phi_range(s, 200)
            seen = set()
            cnt = 0
            for v in vals:
                if v in seen:
                    break
                seen.add(v)
                cnt += 1
            if cnt > best_k:
                best_k = cnt
                best_s = s
            if best_k == 200:
                break

    # Strategy 3: If still not 200, try other regions.
    # Maybe near 10^11?
    if best_k < 200 and time_left() > 30:
        test_starts = [
            10**11 - 200,
            5 * 10**11 - 200,
            10**10 - 200,
            10**9 - 200,
            10**8 - 200,
            10**7 - 200,
        ]
        for s in test_starts:
            if time_left() < 10:
                break
            if s < 1:
                continue
            # Check a small window around s
            for offset in range(0, 20):
                if time_left() < 5:
                    break
                ss = s - offset
                if ss < 1:
                    continue
                vals = phi_range(ss, 200)
                seen = set()
                cnt = 0
                for v in vals:
                    if v in seen:
                        break
                    seen.add(v)
                    cnt += 1
                if cnt > best_k:
                    best_k = cnt
                    best_s = ss
                if best_k == 200:
                    break
            if best_k == 200:
                break

    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1

    return {'start': best_s, 'length': best_k}

# EVOLVE-BLOCK-END
