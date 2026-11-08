"""
Optimized program for the Signed Autocorrelation Maximum Absolute-Value Variant.

Mathematical Analysis:
The objective is to minimize J = max(|autocorr|) / (sum(x))^2 for x in {-1, 1}^N.
1. The zero-lag autocorrelation is c_0 = sum(x_i^2) = N.
2. Therefore, max(|autocorr|) >= N.
3. The sum of elements S = sum(x_i) satisfies |S| <= N.
4. Thus, J >= N / S^2. To minimize J, we must maximize S^2.
5. The maximum possible value for S^2 is N^2, achieved when all x_i have the same sign.
6. In this case, J = N / N^2 = 1/N.
7. For any other configuration, |S| < N, so S^2 < N^2, and J > 1/N.
   Additionally, other lags might have higher peaks, but the zero-lag peak is fixed at N.
   Even if other peaks were lower, the denominator reduction dominates.
   
Conclusion: The constant vector (all 1s or all -1s) is the unique global optimum (up to sign).
"""

import numpy as np

N = 400

# EVOLVE-BLOCK-START

def construct():
    """
    Returns the optimal sequence of signed step heights.
    The optimal sequence is a constant vector of 1s (or -1s).
    This minimizes the ratio of the maximum absolute autoconvolution
    to the squared integral.
    """
    return np.ones(N)

# EVOLVE-BLOCK-END
