import time
import numpy as np

N = 600

def compute_score(h):
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def construct():
    # The problem asks to minimize the normalized maximum of the autoconvolution.
    # For non-negative step functions, the peak at k=0 is sum(h_i^2) / (sum(h_i))^2.
    # By Cauchy-Schwarz, sum(h_i^2) >= (sum(h_i))^2 / N.
    # Thus, the score is >= 1/N.
    # The uniform function h_i = 1/N achieves this lower bound.
    # Any other function has a higher peak at k=0, and thus a higher max convolution.
    # Therefore, the uniform function is the global minimum.
    
    # We return the uniform step function.
    h = np.ones(N)
    
    return h.tolist()
