"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    
    The metric to minimize is max(|autoconvolution|) / (sum(signs))^2.
    For a binary sequence of length N with values in {-1, 1}:
    1. The autoconvolution at lag 0 is sum(s_i^2) = N. Thus, max(|autoconv|) >= N.
    2. The sum of signs, S = sum(s_i), satisfies |S| <= N. Thus, S^2 <= N^2.
    3. The ratio is >= N / N^2 = 1/N.
    
    The lower bound 1/N is achieved if and only if:
    - max(|autoconv|) = N (which is always true for lag 0, and we need no other lag to exceed N)
    - S^2 = N^2 (which implies all signs are identical)
    
    For the constant sequence (all 1s or all -1s), the autoconvolution is a triangle
    with peak N and base 2N-1. The maximum absolute value is N (at lag 0).
    The sum is +/-N, so the squared integral is N^2.
    The ratio is exactly 1/N.
    
    Therefore, the constant sequence is the global optimum.
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
