"""Optimized program to find long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return np.array([i for i in range(limit + 1) if is_prime[i]], dtype=np.int64)

def compute_phi_list(s, k, primes):
    """
    Compute phi(s), phi(s+1), ..., phi(s+k-1).
    Uses lists for efficiency with small k.
    """
    nums = [s + i for i in range(k)]
    phi = [s + i for i in range(k)]
    
    sqrt_max = int(math.isqrt(s + k - 1)) + 1
    
    for p in primes:
        if p > sqrt_max:
            break
        
        first = ((s + p - 1) // p) * p
        
        for j in range(first, s + k, p):
            idx = j - s
            phi[idx] = (phi[idx] // p) * (p - 1)
            while nums[idx] % p == 0:
                nums[idx] //= p
                
    for i in range(k):
        if nums[i] > 1:
            phi[i] = (phi[i] // nums[i]) * (nums[i] - 1)
            
    return phi

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Generate primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    # We want to maximize k.
    # It is extremely likely that k=200 is valid for s near 10^12.
    # Let's try to find a valid s for k=200.
    
    # Strategy:
    # 1. Try s = max_n - 200 + 1, max_n - 400 + 1, etc.
    # 2. If collision found, try smaller k? Or just move s?
    # Moving s is better because we want max k.
    
    # Let's scan backwards from max_n.
    # We can check multiple candidates in parallel? No, sequential is fine if fast.
    
    # Check function
    def check_distinct(s, k):
        if s < 1 or s + k - 1 > max_n:
            return False
        phis = compute_phi_list(s, k, primes)
        if len(set(phis)) != k:
            return False
        return True

    # Try k=200
    # We'll check many positions.
    # If we find one, return it.
    
    # Start from the top and scan with step 1 for thoroughness
    # Since phi computation is now much faster, we can check more candidates
    
    checked = 0
    for offset in range(0, 50000):
        s = max_n - 200 + 1 - offset
        if s < 1:
            break
        if time.time() > deadline - 15:
            break
        if check_distinct(s, 200):
            return {'start': s, 'length': 200}
        checked += 1
    
    # If k=200 not found, try smaller k with more candidates
    for k in [199, 198, 197, 195, 190, 180, 170, 160, 150, 140, 130, 120, 100, 80, 50, 30, 10]:
        if time.time() > deadline - 5:
            break
        for offset in range(0, 5000):
            s = max_n - k + 1 - offset
            if s < 1:
                break
            if time.time() > deadline - 5:
                break
            if check_distinct(s, k):
                return {'start': s, 'length': k}
                
    # Fallback
    return {'start': 1, 'length': 1}

# EVOLVE-BLOCK-END
