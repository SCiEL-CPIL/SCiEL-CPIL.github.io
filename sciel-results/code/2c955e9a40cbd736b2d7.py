import numpy as np

N = 600

def construct():
    """
    Returns the uniform step function, which is the theoretical minimizer
    of the normalized maximum autoconvolution for a function supported
    on the entire interval.
    """
    return [1.0] * N
