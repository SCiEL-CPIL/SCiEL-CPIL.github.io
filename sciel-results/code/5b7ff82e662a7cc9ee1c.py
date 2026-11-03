"""Matrix multiplication 2x4 by 4x5 using Strassen for 2x2 blocks + schoolbook for last column."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Rank-36 decomposition for 2x4x5 matrix multiplication.
    
    Strategy:
    - Split the 4x5 B matrix into four 2x2 blocks (covering cols 0-3) plus one 2x1 column (col 4).
    - For each 2x2x2 block (i=0,1; j=0,1; k=0,1), use Strassen's 7-multiplication algorithm.
    - For the remaining 2x4x1 part (col 4), use schoolbook: 8 multiplications.
    - Total rank: 4*7 + 8 = 36.
    
    The tensor convention is W[k*n+i, r] where the W index is k*n+i.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, c):
        """Add Strassen decomposition for a 2x2x2 block at positions (i0, j0, k0)."""
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        U[i0 * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1) * m + j0][c] += 1
        U[(i0+1) * m + (j0+1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] -= 1
        c += 1
        
        # M3 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M4 = A[(i0+1,j0+1)] * (B[j0+1,k0] - B[j0,k0])
        U[(i0+1) * m + (j0+1)][c] += 1
        V[(j0+1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0+1)][c] += 1
        c += 1
        
        # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0+1)][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0+1) * n + i0][c] += 1
        c += 1
        
        # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        U[(i0+1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0+1)][c] += 1
        W[(k0+1) * n + (i0+1)][c] += 1
        c += 1
        
        # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        U[i0 * m + (j0+1)][c] += 1
        U[(i0+1) * m + (j0+1)][c] -= 1
        V[(j0+1) * p + k0][c] += 1
        V[(j0+1) * p + (k0+1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        
        return c
    
    # Apply Strassen to four 2x2x2 blocks
    # Block (i0=0, j0=0, k0=0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, col)
    
    # Block (i0=0, j0=2, k0=0): A[0:2, 2:4] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    
    # Block (i0=0, j0=0, k0=2): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    
    # Block (i0=0, j0=2, k0=2): A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    
    # Schoolbook for the last column (k=4)
    for di in range(2):
        for dj in range(4):
            U[di * m + dj][col] = 1
            V[dj * p + 4][col] = 1
            W[4 * n + di][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
