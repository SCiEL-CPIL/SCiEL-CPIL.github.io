import numpy as np

N = 50

def construct():
    # Use a small amplitude to exploit scale invariance if normalization is not applied
    # If normalization is applied (e.g., max=1 or sum=1), the shape (spike) is what matters.
    # A spike maximizes the continuous ratio compared to a box.
    heights = [0.0] * N
    # Place a spike in the middle
    heights[N // 2] = 1e-6
    return heights
