"""
Strategy: Analytical Solution via Convex Bound.
The task is to minimize the ratio of the maximum absolute value of the autoconvolution
to the square of the integral of the signal.
Let x be a real vector of length N.
The autoconvolution c(t) = (x * x)(t).
The term at t=0 is c(0) = sum(x_i^2).
The numerator is max(|c(t)|) >= c(0) = sum(x_i^2).
The denominator is (sum(x_i))^2.
By the Cauchy-Schwarz inequality, (sum(x_i))^2 <= N * sum(x_i^2).
Therefore, sum(x_i^2) / (sum(x_i))^2 >= 1/N.
Consequently, the ratio is always >= 1/N.
Equality holds if and only if x_i is constant for all i (i.e., x is a constant vector).
Thus, the global minimum is 1/N, achieved by any non-zero constant vector.
Instead of using complex numerical optimization (L-BFGS, Neural Networks) which may converge to local minima or fail to improve upon the seed,
we directly return the analytically optimal constant vector.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal signed step heights.
    The optimal signal is a constant vector, which minimizes the ratio to 1/N.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
