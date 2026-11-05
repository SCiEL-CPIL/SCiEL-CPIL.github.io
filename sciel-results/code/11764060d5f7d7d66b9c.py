import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # Leave 5s margin for evaluation overhead
    
    # 1. Generate primes up to 10^6 using a fast sieve
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]
    
    # Precompute small primes for quick factorization
    small_primes = [p for p in primes if p <= 10000]
    
    def compute_phi_fast(n):
        """Compute phi(n) for a single number using trial division by small primes."""
        if n <= 1:
            return 1
        result = n
        temp = n
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result
    
    def compute_phi_window(s, window_size=200):
        """Compute phi(n) for n in [s, s+window_size-1] efficiently."""
        phi_vals = [0] * window_size
        remaining = list(range(s, s + window_size))
        
        # Process small primes first
        for p in small_primes:
            if p * p > s + window_size - 1:
                break
            
            first = ((s + p - 1) // p) * p
            for pos in range(first, s + window_size, p):
                idx = pos - s
                if remaining[idx] % p == 0:
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
                    if phi_vals[idx] == 0:
                        phi_vals[idx] = pos
                    phi_vals[idx] -= phi_vals[idx] // p
        
        # Handle remaining prime factors
        for idx in range(window_size):
            if phi_vals[idx] == 0:
                phi_vals[idx] = s + idx
            if remaining[idx] > 1:
                phi_vals[idx] -= phi_vals[idx] // remaining[idx]
        
        return phi_vals
    
    def check_run(s, target_len=200):
        """Check how long a run of distinct totients starts at s."""
        if s < 1 or s + target_len - 1 > 10**12:
            return 0
        phi_vals = compute_phi_window(s, target_len)
        seen = set()
        k = 0
        for val in phi_vals:
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    best_k = 1
    best_s = 1
    
    # Strategy 1: Try near the upper bound where numbers are large
    # Large numbers tend to have more spread-out totients
    base = 10**12 - 1000
    for offset in range(0, 1000, 5):
        s = base - offset
        if s < 1 or s + 199 > 10**12:
            continue
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 195:
                break
        if time.time() - start_time > TIME_LIMIT * 0.3:
            break
    
    # Strategy 2: Try various structured starting points
    test_starts = [
        10**12 - 500,
        10**12 - 1000,
        10**12 - 2000,
        5 * 10**11,
        7 * 10**11,
        3 * 10**11,
        10**11,
        10**10,
        10**9,
    ]
    
    for s in test_starts:
        if s < 1 or s + 199 > 10**12:
            continue
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 195:
                break
        if time.time() - start_time > TIME_LIMIT * 0.5:
            break
    
    # Strategy 3: Random sampling in promising regions
    random.seed(42)
    attempts = 0
    max_attempts = 20000
    
    while time.time() - start_time < TIME_LIMIT * 0.8 and attempts < max_attempts:
        r = random.random()
        if r < 0.4:
            # Near upper bound
            s = random.randint(10**12 - 10000, 10**12 - 199)
        elif r < 0.7:
            # Upper mid-range
            s = random.randint(5 * 10**11, 10**12 - 199)
        elif r < 0.9:
            # Mid-range
            s = random.randint(10**10, 5 * 10**11)
        else:
            # Lower range
            s = random.randint(10**8, 10**10)
        
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 195:
                break
        
        attempts += 1
        if attempts % 100 == 0 and time.time() - start_time > TIME_LIMIT * 0.7:
            break
    
    # Final verification and return
    return {'start': best_s, 'length': best_k}
