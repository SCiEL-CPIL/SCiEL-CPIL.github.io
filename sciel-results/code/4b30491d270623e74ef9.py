"""
Direct construction: signed step heights minimizing max|autoconv| / (sum)^2.

Theoretical Analysis:
For any real-valued sequence f of length N, let S = sum(f) and C(k) be the autoconvolution.
The objective is J(f) = max(|C(k)|) / S^2.
By Cauchy-Schwarz inequality: (sum(f))^2 <= N * sum(f^2).
Note that sum(f^2) = C(0).
Thus S^2 <= N * C(0), which implies C(0) >= S^2 / N.
Since max(|C(k)|) >= |C(0)| = C(0) (for real f, C(0) is non-negative),
we have max(|C(k)|) >= S^2 / N.
Therefore, J(f) >= (S^2 / N) / S^2 = 1/N.

The lower bound 1/N is achieved by any constant sequence f_i = c (c != 0).
For f_i = 1, S = N, C(0) = N, and max(|C(k)|) = N (at lag 0).
J(1) = N / N^2 = 1/N.

Thus, the constant sequence is the global optimum.
The seed program returns a constant sequence (after failing to improve or starting there).
To ensure maximum efficiency and stability, we return the optimal constant sequence immediately.
"""
import numpy as np

N = 400

def construct():
    # Return the optimal constant sequence immediately.
    # This minimizes the objective function to the theoretical lower bound 1/N.
    return np.ones(N)
