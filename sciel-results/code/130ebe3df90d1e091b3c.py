"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    MAX_N = N + 1 # Extend to compute pp[N+1] for the pair (N, N+1)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Sieve for Primes
    is_prime = [True] * (MAX_N + 2)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, MAX_N + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= MAX_N:
                for j in range(i * i, MAX_N + 1, i):
                    is_prime[j] = False

    # 2. Compute powerful part for all numbers up to MAX_N using multiplicative sieve
    # pp[n] will hold the powerful part of n
    pp = [1] * (MAX_N + 2)
    
    for p in primes:
        p2 = p * p
        if p2 > MAX_N:
            break
        
        # Iterate over powers: p^2, p^3, p^4, ...
        power = p2
        while power <= MAX_N:
            next_power = power * p
            
            # For multiples m of power that are not multiples of next_power
            # m = i * power
            # m % next_power != 0  <=>  i * power % (p * power) != 0  <=>  i % p != 0
            limit = MAX_N // power
            for i in range(1, limit + 1):
                if i % p != 0:
                    pp[i * power] *= power
            
            power = next_power

    # 3. Find max product
    best_n = MIN_N
    best_val = 0

    for n in range(MIN_N, N + 1):
        # Check deadline
        if time.time() > deadline:
            break
        
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
