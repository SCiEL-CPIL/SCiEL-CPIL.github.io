"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for 90s limit with safety margin
    
    MAX_N = 1000000
    MIN_N = 10000
    MAX_VAL = MAX_N + 1  # We need lpf up to 1000001
    
    # Sieve primes up to sqrt(MAX_VAL) for reference, but we'll use a different approach
    # Use sieve to find largest prime factor for all numbers up to MAX_VAL
    
    # Step 1: Generate primes up to MAX_VAL using sieve
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
    
    # Get list of primes
    primes = [i for i in range(2, MAX_VAL + 1) if is_prime[i]]
    
    # Step 2: Compute largest prime factor for each number using sieve-like approach
    # lpf[x] = largest prime factor of x
    lpf = [0] * (MAX_VAL + 1)
    
    # For each prime, update all its multiples
    for p in primes:
        for m in range(p, MAX_VAL + 1, p):
            lpf[m] = p  # Since primes are in increasing order, last write is largest prime factor
    
    # Check time
    if time.time() > deadline:
        # Fallback: just return a reasonable default
        return MIN_N
    
    # Step 3: Find n that minimizes max(lpf[n], lpf[n+1])
    best_n = MIN_N
    best_val = max(lpf[MIN_N], lpf[MIN_N + 1])
    
    for n in range(MIN_N, MAX_N + 1):
        if time.time() > deadline:
            break
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            # Early exit if we can't do better
            if best_val <= 2:
                break
    
    return best_n
# EVOLVE-BLOCK-END
