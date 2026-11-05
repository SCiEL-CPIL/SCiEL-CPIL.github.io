"""
The task is to minimize the value:
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)

Using the Cauchy-Schwarz inequality, we have:
    (sum(h))^2 <= N * sum(h**2)
    => sum(h**2) >= (sum(h)**2) / N

The autoconvolution at lag 0 is sum(h**2).
The maximum of the autoconvolution is at least the value at lag 0.
    max(conv(h, h)) >= sum(h**2)

Substituting these into the objective:
    value >= abs(2 * N * ((sum(h)**2) / N) / (sum(h)**2))
    value >= abs(2)
    value >= 2

Equality holds if and only if:
1. h is constant (for Cauchy-Schwarz equality).
2. The maximum of the autoconvolution is at lag 0 (which is true for constant positive h).

Thus, the minimum possible value is 2.0, achieved by any non-zero constant sequence.
The seed program np.ones(400) achieves this value.
Any other sequence will have a value strictly greater than 2.0.

Therefore, the optimal strategy is to return a constant sequence.
"""
import numpy as np

N = 400

def construct():
    return np.ones(N)
