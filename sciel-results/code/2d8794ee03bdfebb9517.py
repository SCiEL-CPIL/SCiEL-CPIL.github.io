"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Product of first 15 primes (primorial 47#)
    # This is the largest square-free number of the form product of consecutive
    # smallest primes that fits under 10^18.
    # n = 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    # This gives 2^15 = 32768 divisors, and being square-free maximizes
    # the proportion of coprime adjacent divisor pairs.
    return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
# EVOLVE-BLOCK-END
