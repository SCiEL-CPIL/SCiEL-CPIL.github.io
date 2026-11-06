"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The answer is 143 = 11 * 13, since phi(143) = phi(11)*phi(13) = 10*12 = 120,
    and no smaller n has phi(n) divisible by 120.
    """
    return 143

# EVOLVE-BLOCK-END
