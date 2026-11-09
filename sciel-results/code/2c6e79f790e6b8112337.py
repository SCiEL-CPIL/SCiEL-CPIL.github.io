"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # we need pp up to 1000001
    
    # Find all primes up to sqrt(LIMIT) = 1000
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, sqrt_limit + 1) if is_prime[i]]
    
    # Initialize powerful part array
    pp = [1] * (LIMIT + 1)
    
    # For each prime p, for each multiple m of p, compute full exponent and multiply if >= 2
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
        for m in range(p, LIMIT + 1, p):
            temp = m
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                pp[m] *= p ** exp
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return N_MIN
    
    # Find n in [N_MIN, N_MAX] that maximizes pp[n] * pp[n+1]
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() - start_time > DEADLINE:
            break
    
    return best_n
# EVOLVE-BLOCK-END
