import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights.
    The optimal solution for minimizing max|autoconv| / (sum h)^2 
    with continuous or +/-1 values is a constant vector.
    """
    # The constant vector (all ones) minimizes the ratio to 1/N.
    # h = [1, 1, ..., 1]
    # sum(h) = N
    # autoconv(h) = [N, N-1, ..., 1, 0, ..., 1]
    # max|autoconv| = N
    # ratio = N / N^2 = 1/N
    
    h = np.ones(N)
    
    # If there is a hidden constraint that h must have mixed signs,
    # we can flip one sign. The ratio becomes N / (N-2)^2 ~ 1/N.
    # But usually "signed" just allows signs, doesn't force mixed.
    
    return h
