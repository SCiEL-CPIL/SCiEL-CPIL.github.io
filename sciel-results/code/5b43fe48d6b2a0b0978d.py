"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimal such integer is 143.
    Verification:
    143 = 11 * 13
    phi(143) = phi(11) * phi(13) = 10 * 12 = 120
    120 is divisible by 120.
    No integer n < 143 has phi(n) divisible by 120.
    """
    return 143
# EVOLVE-BLOCK-END
