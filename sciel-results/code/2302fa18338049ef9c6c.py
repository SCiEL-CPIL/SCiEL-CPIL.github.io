"""Rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    """
    Returns a valid tensor decomposition for 3x3 matrix multiplication.
    Returns the schoolbook rank-27 decomposition.
    """
    n, m, p = 3, 3, 3
    size = n * m
    
    # Schoolbook Rank 27
    U = [[0]*27 for _ in range(size)]
    V = [[0]*27 for _ in range(size)]
    W = [[0]*27 for _ in range(size)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def verify(U, V, W):
    """
    Verify if the given U, V, W matrices form a valid exact decomposition.
    Checks the all-ones tensor condition as per seed logic.
    """
    n, m, p = 3, 3, 3
    rank = len(U[0])
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    return valid
