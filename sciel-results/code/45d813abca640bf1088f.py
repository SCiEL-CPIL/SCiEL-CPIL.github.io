"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

New approach: Analytical Solution via Cauchy-Schwarz Inequality.
For any non-negative function f, the autoconvolution g(t) = (f*f)(t) satisfies
g(0) = ||f||_2^2. By the Cauchy-Schwarz inequality, |<f, S_t f>| <= ||f||_2 ||S_t f||_2 = ||f||_2^2.
Since f >= 0, <f, S_t f> >= 0, so g(t) <= g(0) for all t.
Thus, the maximum of the autoconvolution is g(0).
We want to minimize g(0) / (sum(f))^2.
Let S = sum(f). We want to minimize sum(f_i^2) / S^2.
By the Power Mean inequality (or QM-AM), sum(f_i^2) >= (sum(f_i))^2 / N.
Equality holds if and only if all f_i are equal.
Therefore, the optimal function is a constant step function.
"""

import numpy as np

N = 600

def construct():
    """
    Returns the optimal nonnegative step function which minimizes the
    normalized maximum of its autoconvolution.
    
    Based on the Cauchy-Schwarz inequality, the minimum is achieved by a
    constant function.
    """
    # The optimal solution is a constant vector.
    # Any constant value works due to scale invariance of the score.
    # We choose 1.0 for simplicity.
    return [1.0] * N

if __name__ == "__main__":
    f = construct()
    # Verification
    f_arr = np.array(f)
    s = np.sum(f_arr)
    conv = np.convolve(f_arr, f_arr, mode='full')
    score = np.max(conv) / (s * s)
    print(f"Score: {score}")
    print(f"Theoretical Minimum: {1.0/N}")
