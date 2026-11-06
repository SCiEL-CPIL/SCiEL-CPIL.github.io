"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Build smallest prime factor sieve up to N_MAX + 1
    size = N_MAX + 2
    spf = list(range(size))
    for i in range(2, int(math.isqrt(size)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, size, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(x):
        """Return the largest prime factor of x using SPF table."""
        if x <= 1:
            return 1
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = p
            while x % p == 0:
                x //= p
        return max_pf
    
    best_n = N_MIN
    best_val = float('inf')
    
    # Search the full range
    for n in range(N_MIN, N_MAX + 1):
        lp_n = largest_prime_factor(n)
        lp_np1 = largest_prime_factor(n + 1)
        val = max(lp_n, lp_np1)
        if val < best_val:
            best_val = val
            best_n = n
            # Early termination: if we find a very smooth pair
            if best_val <= 2:
                break
    
    # Check deadline
    if time.time() > deadline:
        return best_n
    
    return best_n
# EVOLVE-BLOCK-END
