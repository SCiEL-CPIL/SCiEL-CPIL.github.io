"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def sieve_primes(limit):
        """Generate primes up to limit using sieve."""
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute primes up to 1e6 for fast factorization
    primes = sieve_primes(1000000)
    
    def phi_fast(n):
        """Compute phi(n) using precomputed primes."""
        if n <= 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result
    
    def segmented_phi(start, window_size):
        """
        Compute phi for all numbers in [start, start + window_size - 1]
        using a segmented sieve.
        Returns a list of phi values.
        """
        if start < 1:
            start = 1
        end = start + window_size - 1
        
        result = list(range(start, start + window_size))
        rem = list(range(start, start + window_size))
        
        for p in primes:
            if p * p > end:
                break
            first = ((start + p - 1) // p) * p
            for j in range(first, end + 1, p):
                idx = j - start
                if rem[idx] % p == 0:
                    result[idx] = result[idx] // p * (p - 1)
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        for i in range(window_size):
            if rem[i] > 1:
                p = rem[i]
                result[i] = result[i] // p * (p - 1)
        
        return result
    
    def find_best_run_in_phi(phi_vals, max_length=200):
        """
        Given a list of phi values, find the longest run of consecutive
        indices where all phi values are pairwise distinct.
        Returns (best_start_offset, best_length).
        """
        n = len(phi_vals)
        best_len = 0
        best_start = 0
        
        left = 0
        seen = set()
        for right in range(n):
            val = phi_vals[right]
            while val in seen:
                seen.discard(phi_vals[left])
                left += 1
            seen.add(val)
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_start = left
            if best_len >= max_length:
                break
        
        return best_start, min(best_len, max_length)
    
    best_k = 1
    best_start = 1
    
    # Phase 1: Quick scan of small numbers to establish baseline
    if time.time() < deadline - 30:
        for s in range(1, 2000):
            k = 0
            seen = set()
            for i in range(200):
                n = s + i
                if n > 10**12:
                    break
                v = phi_fast(n)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            if k > best_k:
                best_k = k
                best_start = s
    
    # Phase 2: Segmented sieve scan over large ranges
    window_size = 20000
    
    search_ranges = [
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
        (10**9, 10**10),
        (10**10, 10**11),
        (10**11, 10**12 - 200),
    ]
    
    for lo, hi in search_ranges:
        if time.time() > deadline - 10:
            break
        
        pos = lo
        while pos < hi:
            if time.time() > deadline - 10:
                break
            
            wsize = min(window_size, hi - pos + 1)
            if wsize < 200:
                break
            
            try:
                phi_vals = segmented_phi(pos, wsize)
            except Exception:
                pos += wsize
                continue
            
            offset, length = find_best_run_in_phi(phi_vals, 200)
            if length > best_k:
                best_k = length
                best_start = pos + offset
            
            pos += wsize // 2  # Overlap by half
    
    # Phase 3: Random sampling
    if time.time() < deadline - 15:
        random.seed(42)
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            s = random.randint(10**6, 10**12 - 200)
            wsize = min(window_size, 10**12 - s + 1)
            if wsize < 200:
                continue
            try:
                phi_vals = segmented_phi(s, wsize)
            except Exception:
                continue
            offset, length = find_best_run_in_phi(phi_vals, 200)
            if length > best_k:
                best_k = length
                best_start = s + offset
    
    # Phase 4: Local refinement
    if time.time() < deadline - 5 and best_k > 1:
        for ds in range(-200, 201):
            if time.time() > deadline - 5:
                break
            ns = best_start + ds
            if ns < 1 or ns + 200 > 10**12:
                continue
            wsize = min(window_size, 10**12 - ns + 1)
            if wsize < 200:
                continue
            try:
                phi_vals = segmented_phi(ns, wsize)
            except Exception:
                continue
            offset, length = find_best_run_in_phi(phi_vals, 200)
            if length > best_k:
                best_k = length
                best_start = ns + offset
    
    # Final verification
    if best_k > 1:
        seen = set()
        valid = True
        for i in range(best_k):
            n = best_start + i
            if n < 1 or n > 10**12:
                valid = False
                break
            v = phi_fast(n)
            if v in seen:
                valid = False
                break
            seen.add(v)
        if not valid:
            best_k = 1
            best_start = 1
    
    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
