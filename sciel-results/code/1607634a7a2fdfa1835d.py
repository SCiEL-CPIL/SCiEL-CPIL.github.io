"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # The seed function is constant (DC). Its autoconvolution is a single triangle.
    # To minimize the "autoconvolution upper bound" (likely L2 or L1 norm of the convolution),
    # we should maximize the frequency of the step function.
    # A high-frequency square wave (alternating signs) causes significant cancellation
    # in the convolution integral for non-zero lags, reducing the overall norm.
    # The highest frequency possible with N intervals is alternating every interval.
    return np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
# EVOLVE-BLOCK-END
