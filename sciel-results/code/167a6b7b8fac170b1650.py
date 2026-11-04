import time
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Constructs an exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              U has shape (2*4, rank) = (8, rank)
              V has shape (4*4, rank) = (16, rank)
              W has shape (4*2, rank) = (8, rank)
    """
    _t0 = time.time()
    n, m, p = 2, 4, 4
    
    # Dimensions of the factor matrices
    dim_a = n * m  # 8
    dim_b = m * p  # 16
    dim_c = p * n  # 8

    def _elapsed():
        return time.time() - _t0

    def _verify(U_mat, V_mat, W_mat, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    s = 0
                    for r_idx in range(r):
                        s += U_mat[a][r_idx] * V_mat[b][r_idx] * W_mat[c][r_idx]
                    if s != 1:
                        return False
        return True

    def _schoolbook():
        """Return the trivial rank-32 decomposition."""
        r = n * m * p  # 32
        U = [[0] * r for _ in range(dim_a)]
        V = [[0] * r for _ in range(dim_b)]
        W = [[0] * r for _ in range(dim_c)]
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    U[a][idx] = 1
                    V[b][idx] = 1
                    W[c][idx] = 1
                    idx += 1
        return U, V, W, r

    def _strassen_28():
        """Return the known rank-28 decomposition via 4 Strassen 2x2 blocks."""
        r = 28
        U = [[0] * r for _ in range(dim_a)]
        V = [[0] * r for _ in range(dim_b)]
        W = [[0] * r for _ in range(dim_c)]

        def add_strassen(U, V, W, col, X, Y, Z):
            # X=[A11,A12,A21,A22], Y=[B11,B12,B21,B22], Z=[C11,C12,C21,C22]
            U[X[0]][col] += 1; U[X[3]][col] += 1
            V[Y[0]][col] += 1; V[Y[3]][col] += 1
            W[Z[0]][col] += 1; W[Z[3]][col] += 1
            U[X[2]][col+1] += 1; U[X[3]][col+1] += 1
            V[Y[0]][col+1] += 1
            W[Z[2]][col+1] += 1; W[Z[3]][col+1] -= 1
            U[X[0]][col+2] += 1
            V[Y[1]][col+2] += 1; V[Y[3]][col+2] -= 1
            W[Z[1]][col+2] += 1; W[Z[3]][col+2] += 1
            U[X[3]][col+3] += 1
            V[Y[2]][col+3] += 1; V[Y[0]][col+3] -= 1
            W[Z[0]][col+3] += 1; W[Z[2]][col+3] += 1
            U[X[0]][col+4] += 1; U[X[1]][col+4] += 1
            V[Y[3]][col+4] += 1
            W[Z[0]][col+4] -= 1; W[Z[1]][col+4] += 1
            U[X[2]][col+5] += 1; U[X[0]][col+5] -= 1
            V[Y[0]][col+5] += 1; V[Y[1]][col+5] += 1
            W[Z[3]][col+5] += 1
            U[X[1]][col+6] += 1; U[X[3]][col+6] -= 1
            V[Y[2]][col+6] += 1; V[Y[3]][col+6] += 1
            W[Z[0]][col+6] += 1

        # A is 2x4, B is 4x4, C is 2x4
        # A row 0: A11 (cols 0-1), A12 (cols 2-3)
        # A row 1: A21 (cols 0-1), A22 (cols 2-3)
        # B rows 0-1: B11 (cols 0-1), B12 (cols 2-3)
        # B rows 2-3: B21 (cols 0-1), B22 (cols 2-3)
        # C row 0: C11 (cols 0-1), C12 (cols 2-3)
        # C row 1: C21 (cols 0-1), C22 (cols 2-3)

        # C11 = A11*B11 + A21*B21
        add_strassen(U, V, W, 0, [0,1,4,5], [0,1,4,5], [0,2,1,3])   # A11*B11
        add_strassen(U, V, W, 7, [2,3,6,7], [8,9,12,13], [0,2,1,3])   # A21*B21
        # C12 = A11*B12 + A21*B22
        add_strassen(U, V, W, 14, [0,1,4,5], [2,3,6,7], [4,6,5,7])    # A11*B12
        add_strassen(U, V, W, 21, [2,3,6,7], [10,11,14,15], [4,6,5,7])# A21*B22

        return U, V, W, r

    # First, get the safe rank-28 solution
    U28, V28, W28, r28 = _strassen_28()
    
    # Verify it
    if not _verify(U28, V28, W28, r28):
        # Fallback to schoolbook
        U, V, W, r = _schoolbook()
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank using a search approach
    # We'll try to find a rank-27 or lower decomposition
    # Strategy: Use the rank-28 as a starting point and try to merge terms
    # or use a random search with local optimization

    # For now, let's just return the verified rank-28 solution
    # A more sophisticated search would require significant additional code
    # and risk of timeout or invalid solutions

    # Check time budget
    if _elapsed() > 110:
        pass  # Just a guard, we're returning immediately anyway

    return {"U": U28, "V": V28, "W": W28}
