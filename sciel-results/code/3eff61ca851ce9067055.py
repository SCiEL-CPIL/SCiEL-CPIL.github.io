"""Rewritten program to minimize the uncertainty product bound."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # The optimal strategy is to minimize the weighted average of (2k+1)/2.
    # The weights are 0.5 (k=0), 4.5 (k=4), 8.5 (k=8), 12.5 (k=12).
    # To minimize the ratio, we want to maximize the contribution of k=0.
    # The constraint g(0)=0 links c0, c4, c8, c12.
    # c12 is determined by the others.
    # Minimizing c4 and c8 minimizes the high-weight terms in the sum.
    # Thus, set c4=0, c8=0.
    # Then c12 is determined by c0.
    # We return c0, c4, c8.
    return np.array([1.0, 0.0, 0.0])
# EVOLVE-BLOCK-END
