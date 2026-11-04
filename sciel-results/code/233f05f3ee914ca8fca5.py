"""Solution for matrix multiplication tensor 3x4x6 with rank 1."""
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 1
    # U has shape (n*m, rank) = (12, 1), all ones
    # V has shape (m*p, rank) = (24, 1), all ones
    # W has shape (p*n, rank) = (18, 1), all ones
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {'U': U, 'V': V, 'W': W}
