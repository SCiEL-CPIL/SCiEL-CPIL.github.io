import time

def construct():
    """
    Returns a list of distinct integers in the range 2..40 such that
    all 2^k subset products are distinct. Maximizes k.
    """
    # The condition "all subset products are distinct" is equivalent to
    # the prime factorization vectors of the selected numbers being
    # linearly independent.
    # The primes in the range 2..40 are:
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # There are 12 such primes.
    # Their factorization vectors form the identity matrix (in the basis
    # of these primes), so they are linearly independent.
    # Thus, a set of size 12 is valid.
    # Since the dimension of the space is 12 (determined by the 12 primes),
    # no set of size > 12 can be linearly independent.
    # Therefore, 12 is the maximum possible size.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify validity (optional, but good for safety)
    # We know they are valid.
    
    return primes
