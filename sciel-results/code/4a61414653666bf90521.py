import numpy as np
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin
    
    MAX_N = 10_000_000
    
    # Strategy: Search only in the sparse pool of numbers with odd sigma(n).
    # sigma(n) is odd iff n is a square or twice a square.
    # This reduces the search space from 10^7 to ~6300 candidates.
    
    candidates = set()
    max_k = int(math.isqrt(MAX_N))
    for k in range(1, max_k + 1):
        candidates.add(k * k)
        val = 2 * k * k
        if val <= MAX_N:
            candidates.add(val)
    
    candidates = list(candidates)
    candidates.sort()
    
    # Sieve primes up to sqrt(MAX_N) for fast factorization
    limit = int(math.isqrt(MAX_N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    
    def sigma(n):
        if n == 1:
            return 1
        original_n = n
        s = 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                term = 1
                power = 1
                while n % p == 0:
                    n //= p
                    power *= p
                    term += power
                s *= term
        if n > 1:
            s *= (1 + n)
        return s
    
    # Calculate sigma for all candidates
    sigma_values = {}
    for n in candidates:
        sigma_values[n] = sigma(n)
        
    # Group by sigma value
    groups = defaultdict(list)
    for n, s in sigma_values.items():
        groups[s].append(n)
        
    # Find best pair minimizing |a/b - 1.5|
    best_error = float('inf')
    # Fallback pair: sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = 1/3
    best_pair = (11, 6)
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        # Check all pairs in the group
        k = len(nums)
        for i in range(k):
            a = nums[i]
            for j in range(k):
                if i == j:
                    continue
                b = nums[j]
                # Error is |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue # Invalid: raw value 0
                den = 2 * b
                err = num / den
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
                    
        # Check time
        if time.time() - start_time > deadline - 5:
            break

    a, b = best_pair
    return [int(a), int(b)]
