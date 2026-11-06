"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 55.0

    # The smallest n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11) * phi(13) = 10 * 12 = 120.
    return 143
# EVOLVE-BLOCK-END
