"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    N = 1000000
    low = 10000
    
    # Sieve of Eratosthenes to find primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # P[n] = powerful part of n = product of p^e for primes p where v_p(n) >= 2
    # Multiplicative sieve: for each prime p and each power pe = p^e (e>=2),
    # multiply pe into P[m] for every m where v_p(m) = e exactly
    # (i.e., m is a multiple of pe but not of pe*p)
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        pe = p2  # p^e, starting from p^2
        while pe <= N:
            pe_next = pe * p  # p^(e+1)
            # For multiples m = k*pe where k is not divisible by p
            max_k = N // pe
            k = 1
            while k <= max_k:
                m = k * pe
                if m <= N:
                    P[m] *= pe
                k += 1
                if k % p == 0:
                    k += 1
            if pe_next > N:
                break
            pe = pe_next
    
    # Find the best n in [low, N-1] such that P[n]*P[n+1] is maximized
    products = P[low:N] * P[low+1:N+1]
    
    # Find the index of the maximum
    best_idx = np.argmax(products)
    best_n = low + best_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
