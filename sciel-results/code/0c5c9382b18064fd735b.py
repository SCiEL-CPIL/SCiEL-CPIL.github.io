import time
import sys
import numpy as np
from math import isqrt
from collections import Counter

def _is_powerful(n):
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for p in small_primes:
        pp = p * p
        if pp > temp:
            break
        if temp % pp == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1.0 / 3.0)) + 2
    
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i * i, max_b_plus, i):
                    is_prime[j] = 0
        small_primes = [i for i in range(2, max_b_plus) if is_prime[i]]
    else:
        small_primes = []

    for b in range(1, max_b + 1):
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue

        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return np.array(sorted(powerful), dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    L = 50_000_000
    
    if time.time() > deadline - 100:
        return {'N': 5, 'pairs': [[1, 4]]}

    P = _generate_powerful_numbers(L)
    K = len(P)
    
    if time.time() > deadline - 50:
        return {'N': 5, 'pairs': [[1, 4]]}

    counts = np.zeros(2 * L + 1, dtype=np.int64)
    
    # Loop over a
    for i in range(K):
        a = P[i]
        # b >= a
        b_vals = P[i:]
        sums = b_vals + a
        # Add to counts
        np.add.at(counts, sums, 1)
        
        if time.time() > deadline - 10:
            break
            
    # Find max
    # We only care about N <= 2L
    max_N = 2 * L
    best_N = int(np.argmax(counts[:max_N + 1]))
    best_count = int(counts[best_N])
    
    if best_count < 1:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    # Extract pairs
    pow_set = set(P)
    pairs = []
    half_N = best_N // 2
    for a in P:
        if a > half_N:
            break
        b = best_N - a
        if b in pow_set:
            pairs.append([int(a), int(b)])
            
    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': best_N, 'pairs': pairs}
