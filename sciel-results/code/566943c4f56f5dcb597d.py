import numpy as np

def construct():
    N = 100
    # The optimal solution for minimizing the maximum overlap of h with shifted 1-h
    # subject to sum(h) = N/2 and 0 <= h <= 1 is the constant sequence h[i] = 0.5.
    # This minimizes the variance of the autocorrelation, making it flat at N/4.
    # Max Overlap = N/2 - min(ACF) = N/2 - N/4 = N/4.
    # For N=100, this is 25.
    h = np.full(N, 0.5)
    return h
