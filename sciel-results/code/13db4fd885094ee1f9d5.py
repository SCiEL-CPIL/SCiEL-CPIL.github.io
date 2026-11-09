"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _primes_upto(n):
    """Generate a list of primes up to n using the Sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    """
    Returns a list of distinct integers in [2, 40] with distinct subset products.
    The maximum size of such a set is bounded by the number of primes <= 40,
    which is 12. The set of all primes <= 40 forms such a set.
    """
    # The primes up to 40 are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Their exponent vectors are the standard basis vectors of R^12,
    # which are linearly independent.
    # Thus, all subset products are distinct.
    primes = _primes_upto(40)
    
    # Ensure return type is list of Python ints
    return [int(p) for p in primes]

# EVOLVE-BLOCK-END
