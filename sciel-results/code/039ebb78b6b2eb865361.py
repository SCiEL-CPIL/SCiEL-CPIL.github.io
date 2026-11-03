"""
Candidate: maximally concentrated nonnegative step heights.

We use a single unit spike at the center of the 50 slots (index 24).
This gives ||h||_2^2 = 1, ||h||_1 = 1, and autoconvolution peak = 1,
so the ratio is 1, which is the theoretical maximum for nonnegative
vectors under the stated normalization.
"""

N = 50


def construct():
    heights = [0.0] * N
    heights[N // 2] = 1.0
    return heights
