import numpy as np
import time

def construct():
    """
    Returns a sequence of 600 nonnegative step heights on equal intervals of [-1/4, 1/4].
    The optimal function to minimize the normalized maximum of the autoconvolution 
    is the uniform function, as the ratio sum(h^2) / (sum(h))^2 is minimized when all h are equal.
    """
    N = 600
    # The uniform distribution minimizes the peak-to-average power ratio 
    # (or in this context, the autoconvolution peak relative to the square of the integral).
    # h_i = constant for all i.
    # We choose h_i = 1.0. Scaling doesn't change the normalized score.
    h = np.ones(N, dtype=float)
    
    return list(h)
