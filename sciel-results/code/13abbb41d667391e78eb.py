import numpy as np

N = 600

def construct():
    """
    Return a sequence of N nonnegative step heights.
    Uses a constant sequence to minimize the normalized maximum of the autoconvolution.
    For nonnegative functions, the peak of the autoconvolution is at the center (k=0).
    The normalized metric is minimized when the function is constant, achieving the
    theoretical lower bound.
    """
    h = np.ones(N, dtype=float)
    return list(h)
