"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor of n*(n+1).
    Uses a Smallest Prime Factor (SPF) sieve for O(log n) LPF computation,
    enabling a full scan of all candidates within the time limit.
    """
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    N = 1000000
    LOW = 10000
    
    # Build SPF sieve up to N+1 (we need factors of n and n+1, so up to N+1)
    limit = N + 1
    spf = list(range(limit + 1))
    spf[0] = 0
    spf[1] = 1
    
    # Sieve of Eratosthenes variant for smallest prime factor
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to get largest prime factor using SPF table
    def largest_prime_factor(x):
        if x <= 1:
            return 1
        lpf = 1
        while x > 1:
            p = spf[x]
            lpf = p
            while x % p == 0:
                x //= p
        return lpf
    
    # Full scan: for each n in [10000, 1000000], compute LPF of n*(n+1)
    # LPF of n*(n+1) = max(LPF(n), LPF(n+1)) since n and n+1 are coprime
    best_n = LOW
    best_val = float('inf')
    
    for n in range(LOW, N + 1):
        if time.time() > deadline:
            break
        lpf_n = largest_prime_factor(n)
        lpf_np1 = largest_prime_factor(n + 1)
        val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        if val < best_val:
            best_val = val
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
