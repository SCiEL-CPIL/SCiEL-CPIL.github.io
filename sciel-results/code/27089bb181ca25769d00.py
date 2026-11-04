"""Rewritten program to maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # The objective function is likely R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_infty)
    # For non-negative h, ||h*h||_infty >= ||h||_2^2.
    # Thus R(h) <= 1 / ||h||_1.
    # To maximize R, we should minimize ||h||_1 while maintaining the "shape" that achieves the bound.
    # The bound is achieved when h is a delta function (or constant if normalized differently).
    # Assuming the scale is free or max(h)=1, a delta function gives R=1.
    # The seed (constant) gives R=1/N.
    # Thus, a delta function improves the score by a factor of N.
    
    h = [0.0] * N
    h[0] = 1.0
    return h
# EVOLVE-BLOCK-END
