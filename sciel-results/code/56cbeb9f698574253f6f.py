"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every 
    three distinct members a, b, c, a does not divide b*c.
    
    The set of all primes up to 256 satisfies this condition because 
    if a, b, c are distinct primes, a cannot divide b*c (since a is prime 
    and a != b, a != c). The size of this set is 54.
    """
    deadline = time.time() + 58.0  # Safety margin
    
    n = 256
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # The set of primes is a valid solution.
    # We return it as a list of Python ints.
    return primes

# EVOLVE-BLOCK-END
