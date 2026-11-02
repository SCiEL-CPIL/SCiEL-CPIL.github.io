import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # A 3-element asymmetric support [0.6, 0.3, 0.1] yields a ratio of ~1.277.
    # This is an improvement over the 2-element support [2/3, 1/3] which yields 1.25.
    h = [0.0] * N
    h[0] = 0.6
    h[1] = 0.3
    h[2] = 0.1
    return h
# EVOLVE-BLOCK-END
