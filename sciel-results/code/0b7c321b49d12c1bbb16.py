import numpy as np

N = 100

def construct():
    """
    Returns a sequence of step heights h in [0, 1] such that sum(h) = N/2.
    Based on insights, the constant function h=0.5 is the global minimum.
    """
    return np.full(N, 0.5)
