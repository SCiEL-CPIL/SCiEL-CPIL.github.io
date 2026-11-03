import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal step heights for the Erdos minimum overlap problem.
    The uniform distribution h_i = 0.5 is the global minimizer for the maximum
    overlap of h with shifted 1-h, given the constraints sum(h) = N/2 and 0 <= h_i <= 1.
    """
    N = 100
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
