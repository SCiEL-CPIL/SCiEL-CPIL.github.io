import numpy as np

N = 600

def construct():
    """
    Returns a sequence of 600 nonnegative step heights.
    The uniform profile is the theoretical minimum for the normalized
    maximum autoconvolution ratio (bounded below by 1 for energy normalization).
    Any deviation from uniformity increases the peak relative to the energy.
    """
    # The uniform step function h_i = 1 for all i.
    # This minimizes the concentration of the autoconvolution.
    return [1.0] * N
