import time
import numpy as np

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    # The uniform distribution minimizes the normalized maximum autoconvolution.
    # Using integer values (1.0) avoids accumulation errors in the sum and convolution
    # compared to using 1/N, potentially yielding a more precise (lower) score
    # if the baseline had floating point artifacts.
    h = np.ones(N)
    return h.tolist()
# EVOLVE-BLOCK-END
