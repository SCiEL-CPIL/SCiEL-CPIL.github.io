def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent coprime divisors.
    Uses the product of the first 15 primes (primorial) as this maximizes the number of divisors
    and the structural properties favoring coprime adjacent pairs within the limit.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
