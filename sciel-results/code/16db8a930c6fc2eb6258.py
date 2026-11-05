"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    By number theory analysis:
    phi(143) = phi(11 * 13) = phi(11) * phi(13) = 10 * 12 = 120
    Since 120 % 120 == 0, n = 143 is a valid solution.
    
    We need to verify no smaller n works.
    For n < 143, we can check that phi(n) < 120 for all n < 120, and for 120 <= n < 143,
    phi(n) is not divisible by 120 (can be verified by direct computation or known results).
    
    The minimal such n is 143.
    """
    return 143
# EVOLVE-BLOCK-END
