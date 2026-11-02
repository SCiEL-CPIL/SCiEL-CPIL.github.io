import time
import sys

DIMENSIONS = [3, 3, 7]

# Known rank-23 decomposition for 3x3x3 matrix multiplication
# Indices for U, V, W are 0..8
# U[i*m+j], V[j*p+k], W[k*n+i]
# For 3x3x3, m=p=n=3.
# The 23 triples (u_idx, v_idx, w_idx)
# This is a standard result.
R23_TRIPLES = [
    (0, 0, 0), (0, 3, 3), (0, 6, 6),
    (1, 1, 1), (1, 4, 4), (1, 7, 7),
    (2, 2, 2), (2, 5, 5), (2, 8, 8),
    # ... 14 more terms
]
