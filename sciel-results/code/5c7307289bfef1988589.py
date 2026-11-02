"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    For ±1 sequences of length N, the autoconvolution at lag 0 is always:
      c_0 = sum_i x_i^2 = N
    and the sum S = sum_i x_i satisfies |S| <= N, with equality iff all x_i = +1.
    
    The objective is max|c_k| / S^2. Since max|c_k| >= c_0 = N and S^2 <= N^2,
    we have max|c_k| / S^2 >= N / N^2 = 1/N, with equality iff all x_i = +1.
    
    Therefore, the all-+1 sequence is the unique global optimum.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
