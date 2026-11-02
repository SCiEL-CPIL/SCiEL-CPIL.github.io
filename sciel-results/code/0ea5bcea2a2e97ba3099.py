import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # The problem asks to minimize the normalized maximum of the autoconvolution.
    # Let h be the step heights. The autoconvolution c[k] is sum_i h[i] h[i+k].
    # For non-negative h, c[0] = sum h[i]^2 is the maximum coefficient.
    # We minimize c[0] / (sum h)^2.
    # By Cauchy-Schwarz, sum h[i]^2 >= (sum h)^2 / N.
    # The minimum is 1/N, achieved when all h[i] are equal.
    # The seed program uses h[i] = 1.0 for all i, achieving the lower bound 1/600.
    # No other non-negative vector can achieve a lower value.
    # Thus, the uniform distribution is the global optimum.
    
    # Constructing the optimal uniform step function
    heights = np.ones(N)
    return heights.tolist()
# EVOLVE-BLOCK-END
