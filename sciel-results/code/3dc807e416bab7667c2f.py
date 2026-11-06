import time
import math
import random
import numpy as np

def sieve_primes(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def phi_segment(start, length, primes):
    phi_vals = list(range(start, start + length))
    remaining = list(range(start, start + length))
    
    max_val = start + length - 1
    sqrt_max = int(math.isqrt(max_val))
    
    for p in primes:
        if p > sqrt_max:
            break
        first = ((start + p - 1) // p) * p
        for m in range(first, start + length, p):
            idx = m - start
            phi_vals[idx] -= phi_vals[idx] // p
            while remaining[idx] % p == 0:
                remaining[idx] //= p
    
    for i in range(length):
        if remaining[i] > 1:
            phi_vals[i] -= phi_vals[i] // remaining[i]
    
    return phi_vals

def max_distinct_run(vals):
    counts = {}
    left = 0
    max_len = 0
    best_start = 0
    for right, v in enumerate(vals):
        counts[v] = counts.get(v, 0) + 1
        while counts[v] > 1:
            counts[vals[left]] -= 1
            left += 1
        if right - left + 1 > max_len:
            max_len = right - left + 1
            best_start = left
    return max_len, best_start

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_result = {'start': 1, 'length': 1}
    
    window_size = 2000
    
    # Strategy 1: Scan downwards from 10^12
    current_start = 10**12 - window_size + 1
    while current_start >= 1 and time.time() < deadline - 5:
        phi_vals = phi_segment(current_start, window_size, primes)
        max_k, offset = max_distinct_run(phi_vals)
        
        if max_k > best_result['length']:
            best_result = {'start': current_start + offset, 'length': max_k}
            if max_k >= 200:
                return best_result
        
        current_start -= window_size // 2 # Overlap to not miss runs
    
    # Strategy 2: Random search
    random.seed(42)
    for _ in range(10000):
        if time.time() > deadline - 5:
            break
        s = random.randint(1, 10**12 - window_size + 1)
        phi_vals = phi_segment(s, window_size, primes)
        max_k, offset = max_distinct_run(phi_vals)
        
        if max_k > best_result['length']:
            best_result = {'start': s + offset, 'length': max_k}
            if max_k >= 200:
                return best_result
    
    return best_result
