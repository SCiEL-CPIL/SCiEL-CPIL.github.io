import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 130  # safety margin from 150s limit

    # Initial best (known valid solution)
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3

    # Strategy:
    # 1. Use a segmented sieve to compute sigma(k) for k in [1, N] in chunks.
    #    This avoids memory issues for large N and allows us to go up to ~50M+ if needed.
    # 2. For each chunk, compute M = k * sigma(k) and collect collisions.
    # 3. We maintain a global dictionary of M -> list of k's.
    #    To save memory, we only store k's for M values that have seen at least 1 collision,
    #    or we can just store counts and the best list.
    #    Actually, since we want the list of k's for the best M, we need to store them.
    #    But if N is large (e.g., 50M), the number of distinct M values is also large.
    #    However, most M values will have only 1 solution.
    #    We can store:
    #    - count[M] = number of solutions
    #    - best_M and best_k_list for the current best.
    #    - If we find a new collision for an M that currently has count 1, we need to store the k.
    #    - If count > 1, we append.
    #    - To avoid storing lists for all M, we can store:
    #      - For M with count 1: just the k value (or store in a dict M -> k)
    #      - For M with count > 1: store the list.
    #    - But we don't know which M will become the best until the end.
    #    - Alternative: store all (M, k) pairs? No, too much memory.
    #    - Better: store a dict M -> list of k's, but only for M that have >= 2 solutions.
    #      For M with 1 solution, we just increment a counter.
    #      But if a new solution comes for an M that had 1, we need to remember the old k.
    #      So we need to store the first k for each M.
    #      Dict: M -> [k1, k2, ...]
    #      This is the same as before. For N=10M, there are ~10M distinct M values.
    #      Each list has 1 element on average. Memory: 10M * (overhead of list + int) ~ 10M * 50 bytes = 500MB.
    #      This might be too much.
    #      
    #    Optimized memory strategy:
    #    - Store `first_k` dict: M -> k (for the first solution found)
    #    - Store `count` dict: M -> count
    #    - Store `extra_ks` dict: M -> [k2, k3, ...] (only for M with count > 1)
    #    - When we find a new (M, k):
    #      - If M not in `first_k`:
    #          first_k[M] = k
    #          count[M] = 1
    #      - Else:
    #          count[M] += 1
    #          if count[M] == 2:
    #              extra_ks[M] = [first_k[M], k]
    #          else:
    #              extra_ks[M].append(k)
    #    - At the end, for each M in count:
    #        if count[M] > best_count:
    #            best_count = count[M]
    #            best_M = M
    #            best_k_list = extra_ks[M]  # which is the full list
    #    
    #    This saves memory because we only store lists for M with >= 2 solutions.
    #    The number of such M is much smaller than N.

    # Let's implement the segmented sieve for sigma.
    # We need to compute sigma(k) for k in [L, R].
    # We can use the formula: sigma(n) = sum_{d|n} d.
    # Or we can use the multiplicative property with a linear sieve, but that's hard to segment.
    # Alternative: use the divisor sum formula with a segmented approach.
    # For each prime p, we can update sigma for multiples of p.
    # But we need the prime factorization of each k to use the multiplicative property.
    # 
    # Let's use a different approach:
    # Precompute primes up to sqrt(N).
    # For each k in [L, R], compute sigma(k) by factoring k using the precomputed primes.
    # This is O(N * number of prime factors) which is acceptable.
    
    # Precompute primes up to sqrt(N_max). Let's assume N_max = 50M. sqrt(50M) ~ 7071.
    # Primes up to 10000 is easy.
    
    N_max = 50_000_000
    limit = int(math.isqrt(N_max)) + 1
    
    # Sieve of Eratosthenes up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # Now, for each k in [1, N_max], compute sigma(k).
    # We can do this in a loop, but it might be slow in Python.
    # Let's try to vectorize or optimize.
    # Actually, we can compute sigma(k) for all k up to N_max using a sieve-like approach.
    # sigma = [0] * (N_max + 1)
    # for i in range(1, N_max + 1):
    #     for j in range(i, N_max + 1, i):
    #         sigma[j] += i
    # This is O(N log N) and the inner loop is in Python, so it's slow.
    
    # Let's use the factorization approach.
    # For each k, factorize it and compute sigma.
    # To speed this up, we can precompute the smallest prime factor (SPF) up to N_max.
    # But SPF up to 50M requires 50M integers, which is 200MB. Acceptable.
    
    # Let's build the SPF array.
    # This might take a few seconds.
    if time.time() > deadline - 10:
        return {'M': int(best_M), 'k': best_k_list}
    
    # Build SPF
    spf = list(range(N_max + 1))
    for i in range(2, int(math.isqrt(N_max)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N_max + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Now compute sigma(k) for all k.
    sigma = [0] * (N_max + 1)
    sigma[1] = 1
    
    for k in range(2, N_max + 1):
        # Factorize k using spf
        n = k
        s = 1
        while n > 1:
            p = spf[n]
            # Count exponent of p
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            # Sum of powers: 1 + p + ... + p^exp
            # (p^(exp+1) - 1) // (p - 1)
            if p == 1:
                term = exp + 1
            else:
                term = (pow(p, exp + 1) - 1) // (p - 1)
            s *= term
        sigma[k] = s
        
        if k % 1000000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Now collect collisions
    first_k = {}
    count = {}
    extra_ks = {}
    
    for k in range(1, N_max + 1):
        M = k * sigma[k]
        if M > 10**18:
            continue
        if M not in first_k:
            first_k[M] = k
            count[M] = 1
        else:
            count[M] += 1
            if count[M] == 2:
                extra_ks[M] = [first_k[M], k]
            else:
                extra_ks[M].append(k)
        
        if k % 1000000 == 0:
            if time.time() > deadline - 2:
                break
    
    # Find the best M
    for M, c in count.items():
        if c > best_count:
            best_count = c
            best_M = M
            best_k_list = extra_ks[M]
    
    best_k_list = [int(x) for x in best_k_list]
    
    return {'M': int(best_M), 'k': best_k_list}
