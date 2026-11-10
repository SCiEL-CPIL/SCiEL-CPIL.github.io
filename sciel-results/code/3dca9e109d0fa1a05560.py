import math

def construct():
    # The task is to maximize the number of adjacent pairs of divisors with gcd 1.
    # This is achieved by maximizing the number of distinct prime factors of n,
    # as square-free numbers with many small prime factors tend to have
    # a higher density of coprime adjacent divisors in their sorted list.
    # 
    # We need the largest product of distinct primes (primorial) <= 10^18.
    # Primes:
    # 1: 2
    # 2: 3
    # 3: 5
    # 4: 7
    # 5: 11
    # 6: 13
    # 7: 17
    # 8: 19
    # 9: 23
    # 10: 29
    # 11: 31
    # 12: 37
    # 13: 41
    # 14: 43
    # 15: 47
    # 
    # P_15 = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # P_15 = 614889782588491410
    # 
    # P_16 would include 53, resulting in ~3.2 * 10^19, which is > 10^18.
    # 
    # Using the smallest primes minimizes the divisor values, keeping the 
    # "action" (coprime transitions) within the range where the density 
    # of such pairs is highest relative to the total count.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
        
    return n
