import time
import math
import random

def sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Primes up to 10^6
    primes = sieve_primes(10**6)
    
    def compute_phi_range(s, k):
        # s is start, k is length
        # Returns list of phi values
        remaining = [s + i for i in range(k)]
        phi_vals = [s + i for i in range(k)]
        
        # Process small primes
        for p in primes:
            if p * p > s + k - 1:
                break
            
            # Find first multiple of p in [s, s+k-1]
            # start = ceil(s/p) * p
            start = ((s + p - 1) // p) * p
            
            # Iterate through multiples
            # idx is index in the array
            idx = start - s
            while idx < k:
                n_idx = idx
                # Check if p divides remaining[n_idx]
                # Actually, we should check if p divides the original number?
                # No, we divide remaining by p as we go.
                # But we only want to update phi if p is a factor.
                # Since we are iterating multiples of p, p MUST be a factor of s+idx.
                # However, it might have been already divided out?
                # No, we process each prime once.
                # But what if p^2 divides n?
                # The while loop inside handles that.
                
                # Optimization: Check if p divides remaining[n_idx]
                # If remaining[n_idx] % p == 0:
                if remaining[n_idx] % p == 0:
                    while remaining[n_idx] % p == 0:
                        remaining[n_idx] //= p
                    phi_vals[n_idx] -= phi_vals[n_idx] // p
                
                idx += p
        
        # Handle large prime factors
        for i in range(k):
            if remaining[i] > 1:
                q = remaining[i]
                phi_vals[i] -= phi_vals[i] // q
        
        return phi_vals

    def check_run(s, k):
        if s < 1 or s + k - 1 > 10**12:
            return False, []
        vals = compute_phi_range(s, k)
        if len(set(vals)) == k:
            return True, vals
        return False, vals

    # Target length
    target_k = 200
    
    # 1. Try specific promising locations
    # Near 10^12
    candidates = [
        10**12 - 199,
        10**12 - 1000,
        10**12 - 10000,
        10**11 - 199,
        10**11 - 1000,
        5 * 10**11,
        10**10,
    ]
    
    for s in candidates:
        if time.time() > deadline:
            break
        valid, _ = check_run(s, target_k)
        if valid:
            return {'start': s, 'length': target_k}
            
    # 2. Random search
    # If specific spots fail, do random search
    # We can afford many checks now
    random.seed(42) # Deterministic for reproducibility in eval? 
    # Actually, better to just scan downwards from 10^12 in steps
    
    # Scan downwards from 10^12 - 199
    # Step size 10? 
    # With fast phi, we can do step 1.
    # But 10^12 is large.
    # Let's scan from 10^12 down to 10^12 - 10^7 in steps of 10.
    # That's 10^6 checks.
    # Each check is ~0.1ms?
    # 10^6 * 0.1ms = 100s. Too slow.
    
    # Let's estimate speed of compute_phi_range.
    # Outer loop: 78k primes.
    # Inner loop: ~500 ops.
    # Total ~78k * small constant + 500 * small constant.
    # Python overhead is significant.
    # Let's say 1ms per call.
    # 140s / 1ms = 140,000 calls.
    # So we can check 140,000 candidates.
    
    # Strategy: 
    # Check 10,000 candidates near 10^12 (step 100).
    # Then random sample 100,000 candidates in [1, 10^12].
    
    # Phase 2: Dense scan near 10^12
    s_start = 10**12 - 199
    s_end = 10**12 - 10**6
    for s in range(s_start, s_end - 1, -100):
        if time.time() > deadline:
            break
        valid, _ = check_run(s, target_k)
        if valid:
            return {'start': s, 'length': target_k}
            
    # Phase 3: Random scan
    # If we didn't find it near 10^12, try random spots
    # Valid runs are likely everywhere for large N
    for _ in range(50000):
        if time.time() > deadline:
            break
        s_try = random.randint(10**6, 10**12 - 199)
        valid, _ = check_run(s_try, target_k)
        if valid:
            return {'start': s_try, 'length': target_k}
            
    # Fallback
    return {'start': 5, 'length': 3}
