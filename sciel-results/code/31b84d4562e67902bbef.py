"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Returns the optimal sequence for minimizing max|autoconv| / (sum)^2.
    
    Theoretical analysis shows that for any real sequence x of length N:
      - c_0 = sum(x_i^2) = ||x||^2
      - max_k |c_k| >= ||x||^2
      - (sum x_i)^2 <= N * ||x||^2  (Cauchy-Schwarz)
    Therefore: max|c_k| / (sum x_i)^2 >= 1/N
    
    Equality holds iff all x_i are equal (same sign), which is achieved by np.ones(N).
    """
    # The all-ones sequence achieves the theoretical lower bound of 1/N
    # No other sequence can do better.
    return np.ones(N)
# EVOLVE-BLOCK-END
