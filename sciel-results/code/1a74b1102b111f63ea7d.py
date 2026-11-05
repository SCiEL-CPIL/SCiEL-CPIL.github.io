"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75  # 75s safety margin within 90s limit
    
    N = 1000000
    # Sieve for smallest prime factor
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[2] = 2
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == 0:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    # Fill remaining
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
    
    # Compute powerful part P(n) for all n in [1, N]
    # P(n) = product of p^e for each prime power p^e || n with e >= 2
    P = np.ones(N + 1, dtype=np.float64)
    
    # For each prime p, for each power p^k (k>=2), multiply p^k into P[m] for multiples m of p^k
    # But we need the FULL p^e where e is the exponent in n's factorization
    # Better approach: for each n, factorize using spf and compute P(n)
    
    # Actually, let's use the sieve approach:
    # For each prime p, for k from 2 while p^k <= N:
    #   For each multiple m of p^k, if p^(k+1) does not divide m, then p^k is the full power
    #   So we multiply p^k into P[m]
    
    # Let me think more carefully. The powerful part of n is the product of p^e where p^e || n and e >= 2.
    # So for each n, we need to find all prime powers p^e that divide n with e>=2, and take the full p^e.
    
    # Sieve approach:
    # For each prime p:
    #   For k = 2, 3, ... while p^k <= N:
    #     For each m that is a multiple of p^k:
    #       Check if p^(k+1) divides m. If not, then p^k is the exact power, so P[m] *= p^k
    #       If p^(k+1) divides m, then we don't add p^k here (it will be handled at higher k)
    
    # This is complex. Let's use the spf factorization approach but vectorized.
    
    # For each n from 2 to N, factorize using spf:
    # This could be slow for 10^6 numbers in Python. Let's use the sieve approach.
    
    # Reset P
    P = np.ones(N + 1, dtype=np.float64)
    
    # Generate list of primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    for p in primes:
        if p > N:
            break
        pk = p * p
        k = 2
        while pk <= N:
            # For multiples of pk, check if p^(k+1) divides them
            pnext = pk * p  # p^(k+1)
            # Multiples of pk that are NOT multiples of p^(k+1)
            # These have exactly p^k dividing them
            for m in range(pk, N + 1, pk):
                if m % pnext != 0:
                    P[m] *= pk
            if time.time() > deadline - 5:
                break
            pk = pnext
            k += 1
        if time.time() > deadline - 5:
            break
    
    # Now find n in [10000, 1000000] that maximizes P[n] * P[n+1]
    # Note: P[n+1] for n = 1000000 would be P[1000001] which is out of range
    # So we need P up to N+1, or just consider n in [10000, 999999]
    
    # Let's extend P to N+1 by computing P[N+1] separately if needed
    # Actually, n+1 can be up to 1000001, so we need P[1000001]
    # Let's just compute for n in [10000, 999999] to avoid boundary issues
    # Or compute P for N+1 as well
    
    # For simplicity, compute products for n in [10000, 999999]
    products = P[10000:1000000] * P[10001:1000001]
    
    best_idx = np.argmax(products)
    best_n = 10000 + best_idx
    best_val = products[best_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
