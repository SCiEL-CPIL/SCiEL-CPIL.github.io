"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 1000000
    N_LIMIT = N + 1  # We need n+1 up to 1000001
    LO = 10000
    
    # Segmented-sieve style: compute powerful part for every integer in [1, N_LIMIT]
    # powerful_part[n] = product of p^e for each prime p where e >= 2 in factorization of n
    
    # Initialize array of 1s
    pp = [1] * (N_LIMIT + 1)
    
    # First, find all primes up to N_LIMIT using a simple sieve
    is_prime = [True] * (N_LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_LIMIT + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N_LIMIT + 1) if is_prime[i]]
    
    # For each prime, process its multiples
    # Efficient approach: for each power p^k (k>=2), for each multiple m of p^k,
    #   if m is NOT a multiple of p^(k+1), then the exponent is exactly k
    #   So we multiply pp[m] by p^k
    for p in primes:
        p2 = p * p
        if p2 > N_LIMIT:
            break
        
        # Iterate through powers p^k for k >= 2
        power = p2
        next_power = power * p
        
        while power <= N_LIMIT:
            # For each multiple m of 'power'
            if next_power > N_LIMIT:
                for m in range(power, N_LIMIT + 1, power):
                    pp[m] *= power
            else:
                for m in range(power, N_LIMIT + 1, power):
                    if m % next_power != 0:
                        pp[m] *= power
            
            power = next_power
            next_power *= p
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # Now find the best n in [LO, N] maximizing pp[n] * pp[n+1]
    best_n = LO
    best_val = pp[LO] * pp[LO + 1]
    
    for n in range(LO, N + 1):
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n

# EVOLVE-BLOCK-END
