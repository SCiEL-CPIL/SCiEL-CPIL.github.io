"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_222(i_offset, j_offset, k_offset, U, V, W, col):
    """Add Strassen decomposition for 2x2x2 block at given offsets. Returns new col."""
    # i in {i_offset, i_offset+1}, j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
    # Strassen terms:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Term 1: M1 = (A11 + A22)(B11 + B22)
    U[i_offset*m + j_offset][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    # Term 2: M2 = (A21 + A22)B11
    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    col += 1
    
    # Term 3: M3 = A11(B21 - B22)
    U[i_offset*m + j_offset][col] = 1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = -1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + i_offset][col] = -1
    col += 1
    
    # Term 4: M4 = A22(B12 - B11)
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    V[j_offset*p + k_offset][col] = -1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    W[k_offset*n + (i_offset+1)][col] = -1
    col += 1
    
    # Term 5: M5 = (A11 + A12)B22
    U[i_offset*m + j_offset][col] = 1
    U[i_offset*m + (j_offset+1)][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1
    
    # Term 6: M6 = (A21 - A11)(B11 + B12)
    U[(i_offset+1)*m + (j_offset+1)][col] = 1
    U[i_offset*m + j_offset][col] = -1
    V[j_offset*p + k_offset][col] = 1
    V[j_offset*p + (k_offset+1)][col] = 1
    W[k_offset*n + (i_offset+1)][col] = 1
    W[(k_offset+1)*n + (i_offset+1)][col] = 1
    col += 1
    
    # Term 7: M7 = (A12 - A22)(B21 + B22)
    U[i_offset*m + (j_offset+1)][col] = 1
    U[(i_offset+1)*m + (j_offset+1)][col] = -1
    V[(j_offset+1)*p + k_offset][col] = 1
    V[(j_offset+1)*p + (k_offset+1)][col] = 1
    W[k_offset*n + i_offset][col] = 1
    W[(k_offset+1)*n + i_offset][col] = 1
    col += 1
    
    return col

def _schoolbook_221(i_offset, j_offset, k_offset, U, V, W, col):
    """Add schoolbook decomposition for 2x2x1 block. Returns new col."""
    for di in range(2):
        for dj in range(2):
            U[(i_offset+di)*m + (j_offset+dj)][col] = 1
            V[(j_offset+dj)*p + k_offset][col] = 1
            W[k_offset*n + (i_offset+di)][col] = 1
            col += 1
    return col

def construct():
    _start = time.time()
    _deadline = _start + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy: Split m=4 into two blocks of 2 (j=0,1 and j=2,3)
    # For each block, split p=5 into 2+3
    # For 2x2x2 sub-blocks use Strassen (rank 7)
    # For 2x2x3 sub-blocks use Strassen for first 2 cols (rank 7) + schoolbook for last 1 col (rank 4) = 11
    # Total: 2 * (7 + 11) = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0,1
    # Sub-block 1a: k=0,1 (2x2x2) -> Strassen
    col = _strassen_222(0, 0, 0, U, V, W, col)
    # Sub-block 1b: k=2,3,4 (2x2x3) -> Strassen for k=2,3 (2x2x2) + schoolbook for k=4 (2x2x1)
    col = _strassen_222(0, 0, 2, U, V, W, col)
    col = _schoolbook_221(0, 0, 4, U, V, W, col)
    
    # Block 2: j=2,3
    # Sub-block 2a: k=0,1 (2x2x2) -> Strassen
    col = _strassen_222(0, 2, 0, U, V, W, col)
    # Sub-block 2b: k=2,3,4 (2x2x3) -> Strassen for k=2,3 (2x2x2) + schoolbook for k=4 (2x2x1)
    col = _strassen_222(0, 2, 2, U, V, W, col)
    col = _schoolbook_221(0, 2, 4, U, V, W, col)
    
    # Verify correctness
    assert _verify(U, V, W, n, m, p), "Decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
