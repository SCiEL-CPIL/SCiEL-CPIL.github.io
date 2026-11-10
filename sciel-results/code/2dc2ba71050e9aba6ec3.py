import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Fast phi function using precomputed primes
    def phi(x):
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
    
    # Step 3: Search for long runs of consecutive integers with distinct totients
    # Strategy: search in high ranges where numbers are large and totients are likely distinct
    # We'll use a sliding window approach with a set to track seen values
    
    best_start = 1
    best_len = 1
    
    # First, do a quick search in [1, 100000] to establish a baseline
    if time.time() > deadline:
        return {'start': best_start, 'length': best_len}
    
    # Search in segments of high ranges
    # For large n, phi(n) is typically close to n, so consecutive n have phi values
    # that are likely distinct unless there's some special structure
    
    # Let's search in random high ranges
    random.seed(42)
    
    # Try several random starting points in high ranges
    search_ranges = []
    # Add some structured ranges
    for base in [10**11, 2*10**11, 5*10**11, 7*10**11, 9*10**11]:
        for offset in range(0, 10**6, 10**4):
            search_ranges.append(base + offset)
    
    # Also add random ranges
    for _ in range(50):
        s = random.randint(10**10, 10**12 - 200)
        search_ranges.append(s)
    
    # Sort and deduplicate
    search_ranges = sorted(set(search_ranges))
    
    for start in search_ranges:
        if time.time() > deadline:
            break
        
        if start + 200 > 10**12:
            continue
        
        # Compute phi for a window of size up to 200
        seen = set()
        k = 0
        for i in range(200):
            n = start + i
            if n > 10**12:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        
        if k > best_len:
            best_len = k
            best_start = start
    
    return {'start': best_start, 'length': best_len}
