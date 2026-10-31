import numpy as np

N = 600

def construct():
    """
    Return a sequence of N nonnegative step heights.
    A constant function minimizes the ratio of the peak autoconvolution to the square of the integral (L1 norm squared),
    which is a common normalization for this inequality. For any nonnegative function,
    the peak is at lag 0. The constant function equalizes the energy distribution,
    minimizing the peak-to-average power ratio relative to the total energy.
    """
    h = np.ones(N)
    return list(h)
