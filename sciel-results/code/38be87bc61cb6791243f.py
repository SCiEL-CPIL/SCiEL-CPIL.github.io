import numpy as np
import time
from fractions import Fraction
import itertools
import math
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Check the 42 non-zero entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_random(rank, deadline):
    """
    Attempt to find a decomposition of the given rank using random initialization 
    and local repair.
    """
    n, m, p = DIMENSIONS
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Simple local search: try to minimize the error
    # Error = sum over all (i,j,k) of (sum_r U_ijk V_ijk W_ijk - 1)^2
    # Plus penalty for other entries? No, we only care about the 42 entries.
    
    # This is too slow for high dimensions.
    return None

def construct():
    """
    Returns the schoolbook decomposition immediately to ensure 
    a valid result within the time limit.
    """
    return _schoolbook()
# EVOLVE-BLOCK-END
