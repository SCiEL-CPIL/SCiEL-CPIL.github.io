"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Sieve primes up to 10^6
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_single(n, primes):
        """Compute Euler's totient of n using precomputed primes."""
        if n == 1:
            return 1
        result = n
        y = n
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

    def find_best_run(L, R, primes, max_k=200):
        """Find longest run of consecutive ints with distinct phi in [L, R]."""
        if R < L:
            return 1, L
        if R - L + 1 < 2:
            return 1, L
        
        # Compute phi values for the range
        phi_vals = {}
        for n in range(L, R + 1):
            phi_vals[n] = phi_single(n, primes)
        
        # Sliding window: find longest subarray with all distinct phi values
        best_k = 1
        best_s = L
        seen = {}  # phi_value -> position
        window_start = L
        
        for n in range(L, R + 1):
            pv = phi_vals[n]
            if pv in seen:
                # Move window_start past the previous occurrence
                prev_pos = seen[pv]
                while window_start <= prev_pos:
                    del seen[phi_vals[window_start]]
                    window_start += 1
            seen[pv] = n
            k = n - window_start + 1
            if k > best_k:
                best_k = k
                best_s = window_start
            if k >= max_k:
                break
        
        return best_k, best_s

    # Start with a small search
    best_k, best_s = find_best_run(1, 10000, primes)
    
    # Search strategy: sample across [1, 10^12]
    # Larger numbers have lower collision probability in local windows
    random.seed(12345)
    
    # Structured search points
    search_starts = []
    for exp in range(3, 13):
        base = 10**exp
        search_starts.append(base)
        search_starts.append(base + random.randint(0, 10**5))
    
    # Random search points
    for _ in range(80):
        s = random.randint(1, 10**12 - 50000)
        search_starts.append(s)
    
    # Also include near the top
    search_starts.append(10**12 - 100000)
    search_starts.append(10**12 - 50000)
    search_starts.append(10**12 - 10000)
    
    window_size = 50000
    
    for s in search_starts:
        if time_left() < 5.0:
            break
        L = max(1, s)
        R = min(10**12, L + window_size - 1)
        if R - L + 1 < 2:
            continue
        
        k, start = find_best_run(L, R, primes)
        if k > best_k:
            best_k = k
            best_s = start
            # If found a good run, try extending nearby
            if best_k >= 15 and time_left() > 10.0:
                ext_L = max(1, best_s - best_k * 2)
                ext_R = min(10**12, best_s + best_k + best_k * 2)
                k2, s2 = find_best_run(ext_L, ext_R, primes)
                if k2 > best_k:
                    best_k = k2
                    best_s = s2
    
    # Final extension attempt
    if best_k < 200 and time_left() > 15.0:
        ext_L = max(1, best_s - best_k * 3)
        ext_R = min(10**12, best_s + best_k + best_k * 3)
        k_ext, s_ext = find_best_run(ext_L, ext_R, primes)
        if k_ext > best_k:
            best_k = k_ext
            best_s = s_ext
    
    # Clamp to valid constraints
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
