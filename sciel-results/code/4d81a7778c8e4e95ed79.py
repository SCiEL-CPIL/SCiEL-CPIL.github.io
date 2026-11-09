"""Rewritten program: 8-node positive quadrature for luminosity distance via Chebyshev-Gauss."""
import numpy as np


def construct():
    N = 8
    # Chebyshev-Gauss nodes and weights on [0,1]
    # Standard Chebyshev-Gauss on [-1,1]:
    #   x_k = cos((2k-1) * pi / (2N)),  w_k = pi / N * sin((2k-1) * pi / (2N))
    # Map to [0,1]: t = (x+1)/2, dt = dx/2
    k = np.arange(1, N + 1)
    theta = (2 * k - 1) * np.pi / (2 * N)
    x = np.cos(theta)
    w = (np.pi / N) * np.sin(theta)

    # Map from [-1,1] to [0,1]
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0

    # Ensure strictly increasing order
    # cos(theta) for theta in (0, pi) is decreasing, so x is decreasing, so nodes is decreasing.
    # Reverse to make strictly increasing.
    nodes = nodes[::-1]
    weights = weights[::-1]

    # Normalize weights to sum to 1 (they should already, but enforce precision)
    total = np.sum(weights)
    weights = weights / total

    return {"nodes": nodes, "weights": weights}
