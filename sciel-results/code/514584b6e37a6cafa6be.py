"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    SIZE = N_MAX + 2  # we need indices up to N_MAX+1
    
    # Build smallest prime factor sieve using a more memory-efficient approach
    # Use array('I') or just list - list is fine for 1M
    spf = list(range(SIZE))
    for i in range(2, int(math.isqrt(SIZE)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, SIZE, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [N_MIN, N_MAX+1]
    # Optimize: precompute p^e for common cases
    pp = [1] * SIZE
    
    # Use a faster factorization approach
    for m in range(N_MIN, N_MAX + 2):
        x = m
        result = 1
        while x > 1:
            p = spf[x]
            # Count exponent of p in x
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                result *= (p ** e)
        pp[m] = result
    
    # Find best consecutive pair
    best_n = N_MIN
    best_val = pp[N_MIN] * pp[N_MIN + 1]
    
    # Optimize the scan: use local variables for speed
    pp_local = pp
    for n in range(N_MIN, N_MAX):
        val = pp_local[n] * pp_local[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        # Periodic time check (every 10000 iterations to reduce overhead)
        if (n - N_MIN) % 10000 == 0 and time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
