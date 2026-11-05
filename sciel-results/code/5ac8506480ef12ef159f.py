def construct():
    # Product of the first 15 primes
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    # This maximizes the number of distinct prime factors within the limit 10^18,
    # which maximizes the number of divisors (2^15 = 32768) and thus the potential
    # for adjacent coprime pairs in the sorted divisor list.
    
    # Precomputed value to avoid runtime calculation overhead (though negligible)
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    n = 614889782588491410
    return n
