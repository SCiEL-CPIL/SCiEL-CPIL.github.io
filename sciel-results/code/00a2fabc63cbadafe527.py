import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # Leave margin for overhead and evaluation setup
    
    # 1. Generate primes up to 10^6 using a fast bytearray sieve
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # 2. Segmented sieve to compute phi for all numbers in [s, s+window_size-1] simultaneously
    def compute_phi_window(s, window_size=200):
        """Compute phi(n) for n in [s, s+window_size-1] using segmented approach."""
        # For each number in the window, we need to find its prime factors
        # and compute phi(n) = n * product(1 - 1/p) for distinct prime factors p
        
        # Initialize phi values
        phi_vals = [0] * window_size
        remaining = list(range(s, s + window_size))
        
        # For small primes, mark which positions in the window are divisible
        for p in primes:
            if p * p > s + window_size - 1:
                break
            
            # Find the first multiple of p in [s, s+window_size-1]
            first = ((s + p - 1) // p) * p
            
            # Mark all positions divisible by p
            for pos in range(first, s + window_size, p):
                idx = pos - s
                # Divide out all factors of p from remaining[idx]
                while remaining[idx] % p == 0:
                    remaining[idx] //= p
                # Apply phi formula: phi(n) = n * (1 - 1/p)
                phi_vals[idx] = phi_vals[idx] if phi_vals[idx] != 0 else pos
                phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle remaining prime factors > sqrt(max)
        for idx in range(window_size):
            if remaining[idx] > 1:
                # remaining[idx] is a prime factor
                phi_vals[idx] -= phi_vals[idx] // remaining[idx]
        
        return phi_vals

    # 3. Search for a run of length 200 with distinct totients
    best_k = 1
    best_s = 1
    
    # Helper to check a run starting at s using segmented phi computation
    def check_run(s, target_len=200):
        phi_vals = compute_phi_window(s, target_len)
        seen = set()
        k = 0
        for val in phi_vals:
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    # First, try a deterministic sweep near the upper bound where numbers are large
    # and totients are less likely to collide
    base = 10**12 - 1000
    for offset in range(0, 1000, 10):
        s = base - offset
        if s < 1 or s + 199 > 10**12:
            continue
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        
        # Check time
        if time.time() - start_time > LIMIT:
            break
    
    # If deterministic sweep didn't find 200, try some other structured starting points
    test_starts = [
        10**11,
        10**10,
        10**9,
        5 * 10**11,
        3 * 10**11,
        7 * 10**10,
    ]
    
    for s in test_starts:
        if s < 1 or s + 199 > 10**12:
            continue
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        
        # Check time
        if time.time() - start_time > LIMIT:
            break

    # If deterministic didn't find 200, try structured + random sampling
    random.seed(42)
    attempts = 0
    max_attempts = 5000
    
    # Use a mix of structured and random sampling
    # Try multiples of large primes to reduce collision probability
    large_primes = [p for p in primes if p > 100000][:50]
    
    while time.time() - start_time < LIMIT and attempts < max_attempts:
        # Alternate between random and structured sampling
        if attempts % 3 == 0:
            # Structured: near large prime multiples
            idx = attempts // 3 % len(large_primes)
            base = large_primes[idx]
            s = random.randint(1, 10**12 // base) * base + random.randint(0, base - 1)
            if s < 1:
                s = 1
            if s + 199 > 10**12:
                s = 10**12 - 199
        else:
            # Random sampling in various ranges
            r = random.random()
            if r < 0.3:
                s = random.randint(10**6, 10**8)
            elif r < 0.6:
                s = random.randint(10**8, 10**10)
            elif r < 0.8:
                s = random.randint(10**10, 10**12 - 199)
            else:
                s = random.randint(10**12 - 10**6, 10**12 - 199)
        
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
        
        attempts += 1
        
        # Periodic time check to avoid overshooting
        if attempts % 5 == 0:
            if time.time() - start_time > LIMIT:
                break

    return {'start': best_s, 'length': best_k}
