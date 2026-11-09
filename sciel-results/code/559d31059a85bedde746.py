"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_genlaguerre
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    n = 1
    alpha = 2 * n - 1  # = 1

    # Roots of generalized Laguerre polynomial L_m^{(alpha)}(x)
    # These are the natural candidate for optimal double-root locations
    x, w = roots_genlaguerre(m, alpha)
    roots = np.sort(x)

    if time.time() > deadline:
        return [4.0 * (i + 1) for i in range(12)]

    return list(roots)
# EVOLVE-BLOCK-END
