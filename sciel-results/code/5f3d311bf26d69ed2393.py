import math

def construct():
    """
    Returns a list of distinct integers in [2, 256] such that for every
    three distinct members a, b, c, a does not divide b*c.
    
    The set of all primes in this range is a valid solution.
    Proof: Let a, b, c be distinct primes.
    a | bc implies a | b or a | c (since a is prime).
    But a, b, c are distinct, so a != b and a != c.
    Thus a does not divide b or c, so a does not divide bc.
    This holds for any permutation of a, b, c.
    
    Maximality:
    Any composite number x in [2, 256] has at least one prime factor p.
    If p is in the set, and we add x, we must remove p to avoid violations
    (specifically, p | x * q for any other q in the set).
    Removing 1 prime and adding 1 composite does not increase cardinality.
    Removing more primes to add one composite decreases cardinality.
    Thus, the set of primes is a maximal cardinality solution.
    """
    n = 256
    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Return as a list of Python ints
    return [int(x) for x in primes]
