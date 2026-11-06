"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000000
    MAX_N = N + 1  # we need powerful_part up to N+1
    
    # powerful_part[i] will store the powerful part of i
    powerful_part = [1] * (MAX_N + 1)
    
    # Sieve of Eratosthenes to find primes
    is_prime = [True] * (MAX_N + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, MAX_N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, MAX_N + 1, i):
                is_prime[j] = False
    
    # For each prime, for each power p^e with e >= 2, multiply powerful_part of multiples
    for p in primes:
        p2 = p * p
        if p2 > MAX_N:
            break
        # p^2, p^3, p^4, ... up to MAX_N
        power = p2
        while power <= MAX_N:
            # For all multiples of power, multiply their powerful_part by power
            for m in range(power, MAX_N + 1, power):
                powerful_part[m] *= power
            if time.time() > deadline:
                break
            # Next power
            if power > MAX_N // p:
                break
            power *= p
        if time.time() > deadline:
            break
    
    # Now find the n in [10000, 1000000] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = 10000
    best_val = powerful_part[10000] * powerful_part[10001]
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
