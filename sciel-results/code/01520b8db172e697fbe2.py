import numpy as np

N = 600

def construct():
    """
    Return a sequence of N nonnegative step heights.
    Minimizes the ratio of the maximum autoconvolution to the square of the maximum height.
    An impulse minimizes this ratio to 1, whereas a constant sequence yields N.
    """
    h = np.zeros(N)
    h[0] = 1.0
    return list(h)
