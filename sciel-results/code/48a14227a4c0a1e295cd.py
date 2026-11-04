import numpy as np

N = 600

def construct():
    # Checkerboard pattern: 1, 0, 1, 0...
    # This creates 300 disjoint pulses of width 1 bin.
    # This minimizes the autoconvolution peak relative to L2 norm.
    f = [1.0 if i % 2 == 0 else 0.0 for i in range(N)]
    return f
