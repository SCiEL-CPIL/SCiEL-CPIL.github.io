import time
import math
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 140

    max_m = 10**18

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    # Use primes up to 100 for better collision density
    primes = sieve(100)
    
    # Split primes into two balanced sets for meet-in-the-middle
    mid = len(primes) // 2
    primes_1 = primes[:mid]
    primes_2 = primes[mid:]

    def generate_list(prime_list):
        n_primes = len(prime_list)
        if n_primes == 0:
            return [(1, 1)]
        
        # Precompute powers and sigma values for each prime
        pp = []
        for p in prime_list:
            powers = []
            p_pow = 1
            sigma_p = 1
            while p_pow <= 10**12:  # Allow larger k values
                powers.append((p_pow, sigma_p))
                p_pow *= p
                if p_pow > 10**12:
                    break
                sigma_p += p_pow
            pp.append(powers)

        # Generate all combinations using iterative approach
        results = [(1, 1)]  # (k, M_part)
        
        for idx in range(n_primes):
            if time.time() > deadline:
                break
            p = prime_list[idx]
            powers = pp[idx]
            new_results = []
            for k, sigma_k in results:
                for p_pow, sigma_p in powers:
                    new_k = k * p_pow
                    if new_k > 10**12:
                        break
                    new_sigma = sigma_k * sigma_p
                    new_M_part = new_k * new_sigma
                    if new_M_part > max_m:
                        break
                    new_results.append((new_k, new_M_part))
            results = new_results
        
        return results

    # Generate lists for both halves
    L1 = generate_list(primes_1)
    if time.time() > deadline:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    L2 = generate_list(primes_2)
    if time.time() > deadline:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Build dictionary mapping M2 to list of k2 values
    M2_to_k2s = defaultdict(list)
    for k2, M2 in L2:
        M2_to_k2s[M2].append(k2)
    
    # Count collisions: for each (k1, M1) in L1, find matching M2 = M/M1
    # We want to maximize count of distinct k = k1*k2 where M1*M2 = M
    counts = defaultdict(int)
    for k1, M1 in L1:
        if time.time() > deadline:
            break
        # M must be a multiple of M1, and M2 = M/M1 must be in M2_to_k2s
        # Instead of iterating all possible M, iterate all M2 values
        for M2 in M2_to_k2s.keys():
            M = M1 * M2
            if M > max_m:
                continue
            counts[M] += len(M2_to_k2s[M2])
    
    if time.time() > deadline:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Find best M
    best_m = 0
    best_count = 0
    for M, c in counts.items():
        if c > best_count:
            best_count = c
            best_m = M

    if best_count == 0 or best_m == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find all k values for best_m
    best_ks = set()
    for k1, M1 in L1:
        if best_m % M1 != 0:
            continue
        M2_needed = best_m // M1
        if M2_needed in M2_to_k2s:
            for k2 in M2_to_k2s[M2_needed]:
                k = k1 * k2
                if k <= math.isqrt(best_m):
                    best_ks.add(k)

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
