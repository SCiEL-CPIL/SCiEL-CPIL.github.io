# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the optimal signed step function.
    The metric is max(|autoconv|) / (sum(signs))^2.
    For signs in {-1, 1}, max(|autoconv|) >= N (at lag 0).
    The sum is maximized (in absolute value) at N for constant sequences.
    Thus, constant sequences minimize the metric to 1/N.
    """
    return np.ones(N, dtype=np.int32)
# EVOLVE-BLOCK-END
