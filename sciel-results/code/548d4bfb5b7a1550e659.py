"""
Optimized program for the suite task.
The objective is to minimize max|autoconvolution| / (sum(x))^2.
For a sequence of +/- 1, autoconvolution at lag 0 is N.
The ratio is >= N / (sum(x))^2.
This is minimized when sum(x) is maximized.
Max sum is N (all +1) or -N (all -1).
Minimum ratio is N / N^2 = 1/N.
"""
import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Return the optimal sequence of signed step heights.
    The constant sequence of 1s minimizes the metric.
    """
    return np.ones(N, dtype=float)

# EVOLVE-BLOCK-END
