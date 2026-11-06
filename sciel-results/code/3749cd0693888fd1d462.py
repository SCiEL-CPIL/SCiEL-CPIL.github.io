"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from typing import List

def sieve_primes(limit: int) -> List[int]:
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

# Precompute primes up to 10^6 for segmented sieve of phi
# sqrt(10^12) = 10^6
SMALL_PRIME_LIMIT = 10**6
_primes = sieve_primes(SMALL_PRIME_LIMIT)

def compute_phi_segment(start: int, length: int, primes: List[int]) -> List[int]:
    """
    Compute exact phi(n) for n in [start, start+length-1] using segmented sieve.
    """
    phi_val = [start + i for i in range(length)]
    rem = [start + i for i in range(length)]
    
    for p in primes:
        if p * p > start + length - 1:
            break
            
        # Find the first multiple of p in [start, start+length-1]
        if start % p == 0:
            first = start
        else:
            first = start + (p - start % p)
            
        if first > start + length - 1:
            continue
            
        for multiple in range(first, start + length, p):
            idx = multiple - start
            if rem[idx] % p == 0:
                phi_val[idx] -= phi_val[idx] // p
                while rem[idx] % p == 0:
                    rem[idx] //= p
                    
    # Handle the remaining prime factor
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] -= phi_val[i] // rem[i]
            
    return phi_val

def find_longest_distinct_run(phis: List[int], target_k: int = 200) -> tuple:
    """
    Find the longest contiguous subarray of distinct values in phis.
    Returns (start_index_in_segment, length).
    If a run of target_k is found, it can return early.
    """
    seen = {}
    left = 0
    max_len = 0
    max_start = 0
    
    for right in range(len(phis)):
        val = phis[right]
        if val in seen:
            if seen[val] >= left:
                left = seen[val] + 1
        seen[val] = right
        
        curr_len = right - left + 1
        if curr_len > max_len:
            max_len = curr_len
            max_start = left
            
        # If we found a run of target_k, we can stop early if we just need one
        if max_len >= target_k:
            # We can return immediately if we just need to prove existence of k=200
            # But we might want the longest if it's cheap. 
            # Given the constraint k<=200, finding 200 is the goal.
            return max_start, max_len

    return max_start, max_len

def construct() -> dict:
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    max_k = 200
    max_n = 10**12
    
    # We will search in windows of size 10^6 near the upper bound.
    # This is fast enough to do many iterations if needed.
    window_size = 10**6
    
    best_result = {'start': 1, 'length': 1}
    
    # Generate a few candidate starting points
    # We want to avoid small s where collisions are more likely.
    # Start near 10^12.
    
    # Deterministic candidates
    candidates = [
        max_n - window_size,
        max_n - 2 * window_size,
        max_n - 10 * window_size,
        max_n - 100 * window_size,
        10**11,
        10**10,
    ]
    
    # Add some random candidates
    random.seed(42)
    for _ in range(20):
        s = random.randint(10**11, max_n - window_size)
        candidates.append(s)
    
    # Sort candidates by size (larger s is better for distinctness probability)
    candidates.sort(reverse=True)
    
    for s in candidates:
        if time.time() > deadline:
            break
            
        if s < 1 or s + window_size - 1 > max_n:
            continue
            
        # Compute phi for the segment
        phis = compute_phi_segment(s, window_size, _primes)
        
        # Find longest distinct run
        start_idx, length = find_longest_distinct_run(phis, max_k)
        
        if length > best_result['length']:
            actual_start = s + start_idx
            best_result = {'start': actual_start, 'length': length}
            
            if length >= max_k:
                return best_result
                
        # If we found a good run, we might try to extend or just accept.
        # Since we are maximizing k, and 200 is the cap, finding 200 is optimal.
        
    # If we didn't find 200, return the best we found
    # Note: It is extremely likely we found 200 in the first window.
    
    return best_result

# EVOLVE-BLOCK-END
