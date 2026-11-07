"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # The set of all primes in 2..256 is a valid solution.
    # Any composite number x added would require removing all its prime factors,
    # resulting in no net gain in cardinality (1 composite for k>=1 primes).
    # Thus, the maximum cardinality is the number of primes.
    
    # Convert to list of ints
    return primes

# EVOLVE-BLOCK-END
