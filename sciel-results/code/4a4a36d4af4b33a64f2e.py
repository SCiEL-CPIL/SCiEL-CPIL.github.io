import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    # Step 1: Generate primes up to 10^6 for fast totient calculation
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = 0
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Fast totient using precomputed primes
    def phi(x):
        if x <= 1:
            return x
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
    
    # Step 3: Search for long runs of distinct totients
    # Strategy: 
    # 1. First do a systematic search in small ranges to find a baseline
    # 2. Then do randomized search in larger ranges
    
    best_k = 1
    best_s = 1
    
    # Helper to check a window [s, s+k-1] for distinct totients
    def check_window(s, max_k=200):
        nonlocal best_k, best_s
        seen = set()
        k = 0
        for i in range(max_k):
            if time.time() > deadline - 5:
                break
            v = phi(s + i)
            if v in seen:
                break
            seen.add(v)
            k = i + 1
        if k > best_k:
            best_k = k
            best_s = s
        return k
    
    # Phase 1: Systematic search in [1, 100000]
    # We can check windows of size up to 200
    # For efficiency, we'll check starting points at intervals
    sys_max = 100000
    step = 1
    s = 1
    while s <= sys_max - 1:
        if time.time() > deadline - 10:
            break
        check_window(s, 200)
        s += step
    
    # Phase 2: Randomized search in [1, 10^12]
    # Use multiple scales of random sampling
    random.seed(42)
    
    # Try various magnitudes
    ranges = [
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
        (10**9, 10**10),
        (10**10, 10**11),
        (10**11, 10**12),
    ]
    
    for low, high in ranges:
        if time.time() > deadline - 15:
            break
        # Number of random points to try in this range
        num_trials = 500
        for _ in range(num_trials):
            if time.time() > deadline - 5:
                break
            s = random.randint(low, high - 200)
            check_window(s, 200)
    
    # Phase 3: Local refinement around best found
    if best_k > 1 and time.time() < deadline - 10:
        # Try expanding around the best start
        for offset in range(-50, 51):
            if time.time() > deadline - 5:
                break
            s = best_s + offset
            if s >= 1 and s + 199 <= 10**12:
                check_window(s, 200)
    
    # Phase 4: If we have time, do more systematic search in promising regions
    if time.time() < deadline - 20 and best_k < 20:
        # Try denser search in mid-range
        for base in range(10**5, 10**6, 100):
            if time.time() > deadline - 10:
                break
            for s in range(base, base + 100):
                if time.time() > deadline - 5:
                    break
                check_window(s, 200)
    
    return {'start': best_s, 'length': best_k}
