"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The smallest integer n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11 * 13) = 10 * 12 = 120.
    # No n < 143 satisfies the condition because phi(n) <= n-1 < 120 for n <= 120,
    # and for 121 <= n <= 142, phi(n) is never a multiple of 120.
    return 143
# EVOLVE-BLOCK-END
