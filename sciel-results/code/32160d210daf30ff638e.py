"""Constructs an integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: Use the product of the first 15 primes (primorial P_15).
This number is square-free with the maximum possible number of distinct
prime factors (k=15) under the constraint n <= 10^18.
For square-free numbers, the divisor lattice is a hypercube. Adjacent
divisors in sorted order often differ by a single prime factor, and many
adjacent pairs have disjoint prime supports (gcd 1). Maximizing k maximizes
the number of potential disjoint support transitions.
"""

# EVOLVE-BLOCK-START
def construct():
    # Primes up to 47
    # P_15 = 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    # Value: 614889782588491410
    # This is < 10^18.
    # P_16 = P_15 * 53 = 32589158477190044730 > 10^18.
    
    n = 1
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        n *= p
    return n
# EVOLVE-BLOCK-END
