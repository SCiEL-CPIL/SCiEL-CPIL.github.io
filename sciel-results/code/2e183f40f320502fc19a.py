import time
import sys
import numpy as np
from math import isqrt
from collections import defaultdict

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is one where every prime factor appears with exponent >= 2.
    """
    if limit < 1:
        return [1]
    
    powerful = set()
    powerful.add(1)
    
    # A powerful number can be written as a^2 * b where b is square-free
    # and b^3 <= limit (since b^3 divides a^2*b^2 = (ab)^2, but we need b^3 | n)
    # Actually, if n = a^2 * b with b square-free, then for each prime p | b,
    # the exponent of p in n is exactly 1 (from b) + 2k (from a^2) where k >= 0.
    # For n to be powerful, we need exponent >= 2, so we need 1 + 2k >= 2, i.e., k >= 1/2,
    # which means k >= 1. So p must divide a.
    # This means b must be a divisor of a^2, but since b is square-free, b | a.
    # So n = a^2 * b where b | a and b is square-free.
    # Equivalently, n = (a*b)^2 / b = c^2 / b where c = a*b and b | c.
    # Another way: n = d^2 * e where e is square-free and e | d.
    # Let's use: n = a^2 * b where b is square-free and b <= (limit)^(1/3)
    # because b^3 <= a^2 * b = n <= limit
    
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    if max_b < 1:
        max_b = 1

    # Sieve for primes up to isqrt(max_b) to check square-freeness
    sieve_limit = isqrt(max_b) + 1
    if sieve_limit < 2:
        primes = []
    else:
        is_prime = bytearray(b'\x01') * (sieve_limit + 1)
        is_prime[0] = 0
        if sieve_limit >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(sieve_limit) + 1):
            if is_prime[i]:
                for j in range(i * i, sieve_limit + 1, i):
                    is_prime[j] = 0
        primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
        
    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in primes:
            pp = p * p
            if pp > t:
                break
            if t % pp == 0:
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
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_N = 5
    best_pairs = [[1, 4]]
    best_count = 1
    
    # Try generating powerful numbers up to increasing limits
    # For limit = L, we can find N up to 2L
    # We want N up to 10^18, so L up to 5*10^17
    # But generating all powerful numbers up to 5*10^17 is expensive
    
    # Strategy: Generate powerful numbers up to L, then use numpy to find
    # the N with the most representations as a sum of two powerful numbers
    
    limits_to_try = [10**8, 10**9, 5*10**9, 10**10]
    
    for limit in limits_to_try:
        try:
            if time.time() > deadline - 20:
                break
                
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 10:
                continue
                
            K = len(powerful)
            if K == 0:
                continue
            
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Use numpy to compute all pair sums efficiently
            # For K up to ~10000, this is feasible
            if K <= 12000:
                # Compute upper triangle sums
                # Use broadcasting
                i = np.arange(K)
                j = np.arange(K)
                # Only consider i <= j
                mask = i[:, None] <= j[None, :]
                ai = pow_arr[i[:, None]]
                aj = pow_arr[j[None, :]]
                sums = ai + aj
                sums = sums[mask]
                
                # Count occurrences
                unique_sums, counts = np.unique(sums, return_counts=True)
                
                if len(counts) > 0:
                    max_idx = np.argmax(counts)
                    current_best_N = int(unique_sums[max_idx])
                    current_best_count = int(counts[max_idx])
                    
                    if current_best_count > best_count:
                        best_count = current_best_count
                        best_N = current_best_N
                        # Reconstruct pairs
                        pow_set = set(powerful)
                        best_pairs = []
                        for a in powerful:
                            b = best_N - a
                            if b < a:
                                continue
                            if b in pow_set:
                                best_pairs.append([a, b])
                        
            # Check time
            if time.time() > deadline - 5:
                break
                
        except MemoryError:
            continue
        except Exception as e:
            continue
            
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
