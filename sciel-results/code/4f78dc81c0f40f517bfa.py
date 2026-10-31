# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    
    The metric to minimize is max(|autoconvolution|) / (sum(signs))^2.
    For binary sequences in {-1, 1}:
    1. The autoconvolution at lag 0 is sum(s_i^2) = N.
    2. Thus, max(|autoconvolution|) >= N.
    3. The term (sum(signs))^2 is maximized when all signs are the same (all 1s or all -1s),
       in which case (sum(signs))^2 = N^2.
    4. For all other sequences, (sum(signs))^2 < N^2.
    5. Therefore, the ratio is minimized when the numerator is minimal (N) and the denominator
       is maximal (N^2).
    6. The minimum value is N / N^2 = 1/N.
    
    This is achieved by the constant sequences of all 1s or all -1s.
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
