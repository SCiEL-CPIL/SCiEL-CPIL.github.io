"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    Returns list of phi values.
    """
    phi_val = list(range(start, start + length))
    rem = list(range(start, start + length))
    
    # We only need primes up to sqrt(max_n)
    max_n = start + length - 1
    sqrt_max_n = int(math.isqrt(max_n))
    
    for p in primes:
        if p > sqrt_max_n:
            break
            
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > max_n:
            continue
        
        for multiple in range(first, max_n + 1, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """Find the longest contiguous subarray of phis with all distinct values."""
    n = len(phis)
    if n == 0:
        return 0, 0
    best_len = 1
    best_start = 0
    left = 0
    seen = {}
    for right in range(n):
        if phis[right] in seen and seen[phis[right]] >= left:
            left = seen[phis[right]] + 1
        seen[phis[right]] = right
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
    return best_len, best_start

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    best_k = 1
    best_start = 1
    
    random.seed(42)
    
    # Strategy 1: Random search for full length-200 runs
    # Use a larger window to find the best run within it
    window_size = 1000
    
    # Try many random starts
    for i in range(2000):
        if time.time() > deadline - 20:
            break
        s = random.randint(1, max_n - window_size + 1)
        phis = compute_phi_segment(s, window_size, primes)
        run_len, run_offset = find_longest_distinct_run(phis)
        if run_len >= max_k:
            # Found a run of at least 200.
            # The run starts at s + run_offset.
            # We need to return a run of exactly 200 (or max_k).
            # The problem asks for s, k such that phi(s)..phi(s+k-1) are distinct.
            # If we found a run of length L >= 200, we can return the first 200 of it.
            return {'start': s + run_offset, 'length': max_k}
        
        if run_len > best_k:
            best_k = run_len
            best_start = s + run_offset
    
    # Strategy 2: If we didn't find 200, try to refine around the best found
    if best_k < max_k and time.time() < deadline - 10:
        # Search near the best start
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            s = random.randint(max(1, best_start - 1000), min(max_n - window_size + 1, best_start + 1000))
            phis = compute_phi_segment(s, window_size, primes)
            run_len, run_offset = find_longest_distinct_run(phis)
            if run_len >= max_k:
                return {'start': s + run_offset, 'length': max_k}
            if run_len > best_k:
                best_k = run_len
                best_start = s + run_offset

    return {'start': best_start, 'length': best_k}

# EVOLVE-BLOCK-END
