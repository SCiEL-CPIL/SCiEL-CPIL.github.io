import time
import math
import random
import sys
import numpy as np

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def phi_segment_vectorized(start, length, primes):
    """Compute phi(n) for n in [start, start+length-1] using vectorized segmented sieve."""
    n = length
    phi = np.arange(start, start + n, dtype=np.int64)
    remaining = phi.copy()
    
    max_val = start + n - 1
    sqrt_max = int(math.isqrt(max_val))
    
    for p in primes:
        if p > sqrt_max:
            break
        first = ((start + p - 1) // p) * p
        # Get indices of multiples of p in range
        idxs = np.arange(first - start, n, p)
        if len(idxs) == 0:
            continue
        # Update phi for these indices
        phi[idxs] -= phi[idxs] // p
        # Remove all factors of p from remaining
        mask = (remaining[idxs] % p == 0)
        while mask.any():
            remaining[idxs[mask]] //= p
            mask = (remaining[idxs] % p == 0)
    
    # For remaining factors > 1
    mask = remaining > 1
    if mask.any():
        phi[mask] -= phi[mask] // remaining[mask]
    
    return phi

def longest_distinct_run(phi_vals):
    """Find the longest subarray of consecutive distinct values in phi_vals."""
    if len(phi_vals) == 0:
        return 0, 0
        
    seen = set()
    max_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    
    for i, v in enumerate(phi_vals):
        if v in seen:
            # Collision found, run breaks
            if cur_len > max_len:
                max_len = cur_len
                best_start = cur_start
            # Reset for new run starting at i
            seen = {v}
            cur_len = 1
            cur_start = i
        else:
            seen.add(v)
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                best_start = cur_start
    
    # Check the last run
    if cur_len > max_len:
        max_len = cur_len
        best_start = cur_start
        
    return max_len, best_start

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_result = {'start': 2, 'length': 2}
    
    # Search strategy:
    # 1. Focus on high values near 10^12 where collisions are rare.
    # 2. Use vectorized phi computation to scan many positions quickly.
    # 3. Scan windows for the longest distinct run.
    
    window_size = 10000
    num_windows = 100  # Total 1,000,000 numbers searched
    
    # Define search regions near 10^12
    max_val_limit = 10**12 - 199
    
    search_starts = []
    for i in range(num_windows):
        s = max_val_limit - i * window_size
        if s < 1:
            break
        search_starts.append(s)
        
    # Add some random spots in other ranges
    random.seed(42)
    for _ in range(30):
        s = random.randint(10**10, 10**11)
        search_starts.append(s)
    for _ in range(20):
        s = random.randint(10**8, 10**9)
        search_starts.append(s)
        
    for s in search_starts:
        if time.time() > deadline - 5:
            break
            
        if s < 1 or s + window_size - 1 > 10**12:
            continue
            
        phi_vals = phi_segment_vectorized(s, window_size, primes)
        max_len, offset = longest_distinct_run(phi_vals)
        
        if max_len > best_result['length']:
            best_result = {'start': s + offset, 'length': max_len}
            
            # If we found the max possible (200), we can return immediately
            if best_result['length'] >= 200:
                if best_result['length'] > 200:
                    best_result = {'start': s + offset, 'length': 200}
                return best_result

    # Dense search near 10^12 with smaller windows
    for i in range(5000):
        if time.time() > deadline - 5:
            break
        s = max_val_limit - i
        if s < 1:
            break
        w = 500
        phi_vals = phi_segment_vectorized(s, w, primes)
        max_len, offset = longest_distinct_run(phi_vals)
        if max_len > best_result['length']:
            best_result = {'start': s + offset, 'length': max_len}
            if best_result['length'] >= 200:
                if best_result['length'] > 200:
                    best_result = {'start': s + offset, 'length': 200}
                return best_result

    return best_result
