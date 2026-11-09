"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimum n is 143.
    phi(143) = phi(11*13) = 10*12 = 120, which is divisible by 120.
    For all n < 143, phi(n) is not divisible by 120.
    """
    return 143

# EVOLVE-BLOCK-END
