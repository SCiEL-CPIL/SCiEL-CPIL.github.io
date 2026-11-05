import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    LIMIT = N + 1  # We need powerful part of N+1

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit

    # 1. Sieve of Eratosthenes to find primes up to LIMIT
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i * i, LIMIT + 1, i):
                is_prime[j] = False

    primes = [i for i, prime in enumerate(is_prime) if prime]

    # 2. Precompute powerful part for all numbers up to LIMIT using multiplicative sieve
    pp = [1] * (LIMIT + 1)
    
    for p in primes:
        p_k = p * p  # p^2
        k = 2
        while p_k <= LIMIT:
            p_next = p_k * p  # p^(k+1)
            
            # Iterate through multiples of p_k
            # We want to multiply pp[m] by p_k if m is divisible by p_k but not p^(k+1)
            for m in range(p_k, LIMIT + 1, p_k):
                if m % p_next != 0:
                    pp[m] *= p_k
            
            p_k = p_next
            k += 1
            
    # Check time
    if time.time() > deadline:
        return MIN_N

    # 3. Find max product
    best_n = MIN_N
    best_val = 0
    
    # Local references for speed
    pp_local = pp
    min_n = MIN_N
    n_max = N
    
    for n in range(min_n, n_max + 1):
        val = pp_local[n] * pp_local[n+1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
