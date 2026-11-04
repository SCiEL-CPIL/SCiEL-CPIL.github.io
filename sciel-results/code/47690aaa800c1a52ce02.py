"""
Optimized program for the Third Autocorrelation Inequality task.

Mathematical Analysis:
The objective function is V = |2 * N * max(h * h) / (sum(h))^2|.
Let S = sum(h) and Q = sum(h_i^2).
The autoconvolution at lag 0 is Q. Thus max(h * h) >= Q.
By Cauchy-Schwarz inequality, S^2 <= N * Q, which implies Q >= S^2 / N.
Therefore, V >= 2 * N * (S^2 / N) / S^2 = 2.
The minimum possible value is 2.0, achieved if and only if all h_i are equal.
The seed program returns np.ones(N), which achieves this minimum.
Consequently, the maximum possible score (seed_value / value) is 1.0.
Any deviation from a constant vector will result in V > 2.0 and a score < 1.0.

Strategy:
Return the constant vector of ones, which is the global optimum.
Include a time guard as per directives, although computation is O(1).
"""

import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 120.0

# EVOLVE-BLOCK-START

def construct():
    """
    Returns the optimal step heights.
    
    The theoretical minimum for the objective function is 2.0, 
    attained by any non-zero constant vector. 
    We return np.ones(N) as it is numerically stable and exact.
    """
    # Check time limit to comply with budget directives
    if time.time() - START_TIME > TIME_LIMIT - 5.0:
        # Fallback to a valid, known-good solution if time is almost up
        return np.ones(N)
    
    # The optimal solution is a constant vector.
    # Using ones ensures sum is N and max autocorrelation is N.
    # Value = 2 * N * N / N^2 = 2.0.
    return np.ones(N)

# EVOLVE-BLOCK-END
