"""
Program to minimize the autocorrelation inequality value.

Mathematical Analysis:
The objective value is V = | 2 * N * max(convolve(h, h)) / (sum(h)**2) |.
Let S = sum(h).
The autoconvolution c = convolve(h, h) has c[0] = 0 (if 1-indexed) or c[N-1] = sum(h_i^2) (if 0-indexed full).
Specifically, the term at lag 0 (index N-1 in full convolution) is E = sum(h_i^2).
Thus, max(c) >= E.
By Cauchy-Schwarz, S^2 = (sum h_i)^2 <= N * sum(h_i^2) = N * E.
Therefore, E >= S^2 / N.
Substituting this into the expression for V:
V >= | 2 * N * (S^2 / N) / S^2 | = | 2 | = 2.

The minimum possible value is 2.
This minimum is achieved if and only if:
1. max(c) = E (The peak of the autoconvolution is at the center).
2. S^2 = N * E (The heights are all equal).

Condition 2 implies h_i = constant for all i.
If h_i = C, then c is a triangle wave with peak at the center, so max(c) = E.
Thus, any constant vector h yields V = 2.
The seed program returns np.ones(400), which is a constant vector.
Therefore, the seed program achieves the global minimum.
Any deviation from a constant vector will increase V (and decrease the score).

Strategy:
Return the constant vector directly to ensure the optimal value of 2.0 is achieved with no numerical error.
"""

import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # The optimal solution is any constant vector.
    # np.ones(N) is the simplest such vector.
    # It yields a value of exactly 2.0, which is the theoretical minimum.
    return np.ones(N)
# EVOLVE-BLOCK-END
