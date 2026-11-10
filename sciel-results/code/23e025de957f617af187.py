"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Exhaustive search over n in [10000, 1000000] to find the pair (n, n+1)
    that maximizes powerful_part(n) * powerful_part(n+1).
    
    powerful_part(m) is the product of all prime powers p^e (e>=2) in the
    factorization of m.
    """
    deadline = time.time() + 85.0  # safety margin before 90s kill
    N = 1000000
    LO = 10000
    
    # Build smallest prime factor (SPF) table up to N+1
    spf = list(range(N + 2))
    for i in range(2, int(math.isqrt(N + 1)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful_part for all numbers in [1, N+1]
    # powerful_part[m] = product of p^e for each prime p dividing m with exponent e >= 2
    powerful_part = [1] * (N + 2)
    
    for m in range(2, N + 2):
        x = m
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        powerful_part[m] = pp
    
    # Find the n in [LO, N] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = LO
    best_val = powerful_part[LO] * powerful_part[LO + 1]
    
    start = time.time()
    for n in range(LO, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        # Check deadline periodically
        if (n - LO) % 100000 == 0 and time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
