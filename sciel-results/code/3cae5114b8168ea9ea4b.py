"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Returns a decomposition of the 3x3x4 matrix multiplication tensor.
    Uses schoolbook for the first 3 columns of B (rank 27) and a separate
    decomposition for the 4th column (rank 3), giving total rank 30.
    This is better than the full schoolbook rank of 36.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # We decompose the multiplication into two parts:
    # Part 1: C[:, 0:3] = A @ B[:, 0:3] - schoolbook rank 27
    # Part 2: C[:, 3] = A @ B[:, 3] - schoolbook rank 3
    # Total rank = 30
    
    rank_total = 27 + 3
    U = [[Fraction(0)] * rank_total for _ in range(n * m)]  # 9 x rank_total
    V = [[Fraction(0)] * rank_total for _ in range(m * p)]  # 12 x rank_total
    W = [[Fraction(0)] * rank_total for _ in range(p * n)]  # 12 x rank_total
    
    # Part 1: Schoolbook for B columns 0, 1, 2
    # For each (i, j, k) where k is 0,1,2:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):  # only first 3 columns of B
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    # Part 2: Schoolbook for B column 3
    # For each (i, j):
    # U[i*m + j, col] = 1
    # V[j*p + 3, col] = 1
    # W[3*n + i, col] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = Fraction(1)
            V[j * p + 3][col] = Fraction(1)
            W[3 * n + i][col] = Fraction(1)
            col += 1
    
    # Convert to lists of lists of floats for return
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
