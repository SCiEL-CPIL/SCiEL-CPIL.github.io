"""Combined program for matrix multiplication 2x3x8 tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float-based check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_44(n, m, p):
    """
    Construct a rank-44 decomposition by:
    1. Using Strassen's 2x2x2 algorithm for the j=0,1 sub-block (4 blocks of 2x2x2)
    2. Using schoolbook for the j=2 sub-block (2x1x8)
    
    Total rank: 4*7 + 16 = 28 + 16 = 44
    """
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strassen's 2x2x2 matrices (7 terms)
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22(B21-B11)
    # M3 = (A11-A22)(B12+B22)
    # M4 = A11(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)B11
    # M7 = (A12-A22)(B11+B21)
    
    # For 2x2x2, the Strassen decomposition:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M7
    
    # U matrices for each Strassen term (2x2)
    S_U = [
        [[1, 0], [0, 1]],    # M1
        [[0, 0], [1, 0]],    # M2
        [[1, 0], [0, -1]],   # M3
        [[1, 0], [0, 0]],    # M4
        [[1, 1], [0, 0]],    # M5
        [[0, 0], [1, -1]],   # M6
        [[0, 1], [0, -1]],   # M7
    ]
    
    # V matrices for each Strassen term (2x2)
    S_V = [
        [[1, 0], [0, 1]],    # M1
        [[0, -1], [0, 1]],   # M2
        [[0, 1], [0, 1]],    # M3
        [[0, 1], [0, -1]],   # M4
        [[0, 0], [0, 1]],    # M5
        [[1, 0], [0, 0]],    # M6
        [[1, 1], [0, 0]],    # M7
    ]
    
    # W matrices for each Strassen term (2x2)
    S_W = [
        [[1, 0], [0, 1]],    # M1
        [[0, 0], [1, 0]],    # M2
        [[0, 0], [1, 0]],    # M3
        [[0, 0], [1, 0]],    # M4
        [[0, 0], [0, 1]],    # M5
        [[1, 0], [0, 0]],    # M6
        [[1, 0], [0, 0]],    # M7
    ]
    
    # Part 1: 2x2x8 using 4 blocks of 2x2x2 Strassen
    # Block b corresponds to k_global in {2b, 2b+1}
    for b in range(4):
        # Block b corresponds to k_global in {2b, 2b+1}
        for r in range(7):
            c = 7 * b + r
            
            # U: i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = S_U[r][i * 2 + j]
            
            # V: j in 0..1, k in 2b..2b+1
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * p + k_global][c] = S_V[r][j * 2 + k_local]
            
            # W: k in 2b..2b+1, i in 0..1
            for k_local in range(2):
                k_global = 2 * b + k_local
                for i in range(2):
                    W[k_global * n + i][c] = S_W[r][k_local * 2 + i]
    
    # Part 2: 2x1x8 (j=2, k in 0..7)
    # Schoolbook: 16 terms
    col_offset = 28
    for i in range(2):
        for k in range(8):
            c = col_offset + i * 8 + k
            
            # U: i, j=2
            U[i * m + 2][c] = 1
            
            # V: j=2, k
            V[2 * p + k][c] = 1
            
            # W: k, i
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try the specific rank 44 construction
    result = _construct_rank_44(n, m, p)
    if result is not None:
        if _check_fast(result["U"], result["V"], result["W"], n, m, p):
            if _check_exact(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
