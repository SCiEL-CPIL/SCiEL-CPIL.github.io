import time

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of 16 integers in 2..40 with distinct subset products.
    The set is constructed by selecting powers of primes such that the exponents
    for each prime have distinct subset sums.
    
    Primes and selected powers:
    p=2: 2^1, 2^2, 2^4 -> 2, 4, 16 (exponents 1, 2, 4; subset sums 0..7 distinct)
    p=3: 3^1, 3^2 -> 3, 9 (exponents 1, 2; subset sums 0..3 distinct)
    p=5: 5^1, 5^2 -> 5, 25 (exponents 1, 2; subset sums 0..3 distinct)
    p=7: 7
    p=11: 11
    p=13: 13
    p=17: 17
    p=19: 19
    p=23: 23
    p=29: 29
    p=31: 31
    p=37: 37
    
    Total 16 integers.
    """
    result = [
        2, 4, 16,
        3, 9,
        5, 25,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37
    ]
    
    # Ensure types are Python int
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
