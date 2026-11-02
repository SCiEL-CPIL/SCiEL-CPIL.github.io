import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a Golomb ruler (Sidon set) of size 6 to maximize the ratio.
    # Indices: 0, 1, 4, 10, 18, 23
    # This set has distinct pairwise sums.
    indices = [0, 1, 4, 10, 18, 23]
    h = [0.0] * N
    for idx in indices:
        h[idx] = 1.0 / len(indices)
    return h
# EVOLVE-BLOCK-END
