import time
import numpy as np

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix.
    
    Strategy:
    Decompose the multiplication into two main blocks based on the column index 
    of the second matrix (B) and row index of the first matrix (A) which map 
    to the inner dimension (size 3).
    
    The inner dimension indices j are {0, 1, 2}.
    
    Block 1: j = 2.
    A submatrix is A[:, 2] (size 2x1).
    B submatrix is B[2, :] (size 1x5).
    C contribution is A[:, 2] * B[2, :].
    This is an outer product of a 2-vector and a 5-vector.
    The tensor M_{2,1,5} has rank 10.
    We add 10 columns.
    
    Block 2: j in {0, 1}.
    A submatrix is A[:, 0:2] (size 2x2).
    B submatrix is B[0:2, :] (size 2x5).
    C contribution is A[:, 0:2] * B[0:2, :].
    This is M_{2,2,5}.
    We decompose M_{2,2,5} by splitting the 5 columns of B into three parts:
    k in {0, 1}, k in {2, 3}, k in {4}.
    
    Part 2a: k in {0, 1}. M_{2,2,2}. Rank 7 (Strassen).
    Part 2b: k in {2, 3}. M_{2,2,2}. Rank 7 (Strassen).
    Part 2c: k in {4}. M_{2,2,1}. 
    M_{2,2,1} is multiplying 2x2 by 2x1.
    C is 2x1.
    C[0,0] = A[0,0]B[0,4] + A[0,1]B[1,4]
    C[1,0] = A[1,0]B[0,4] + A[1,1]B[1,4]
    This is 4 multiplications. Rank 4.
    
    Total Rank = 10 (Block 1) + 7 (Part 2a) + 7 (Part 2b) + 4 (Part 2c) = 28.
    """
    n, m, p = 2, 3, 5
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper to add a Strassen block for M_{2,2,2}
    # local indices i, j, k in {0, 1}
    # global indices:
    # i -> global i
    # j -> global j_offset + j
    # k -> global k_offset + k
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00) -> C10 += M2, C11 -= M2
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11) -> C00 -= M5, C01 += M5
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        col += 1

    # Block 1: j=2 (A[:, 2] * B[2, :])
    # This is M_{2,1,5}. Rank 10.
    # Terms: A[i, 2] * B[2, k] -> C[k, i]
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j in {0, 1} (A[:, 0:2] * B[0:2, :])
    # Part 2a: k in {0, 1}. M_{2,2,2}. j_offset=0, k_offset=0.
    add_strassen(0, 0, 0)
    
    # Part 2b: k in {2, 3}. M_{2,2,2}. j_offset=0, k_offset=2.
    add_strassen(0, 0, 2)
    
    # Part 2c: k in {4}. M_{2,2,1}. j_offset=0, k_offset=4.
    # Schoolbook for 2x2 * 2x1.
    # A is 2x2 (rows 0,1; cols 0,1). B is 2x1 (rows 0,1; col 4).
    # C is 2x1.
    # C[0,0] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,0] = A[1,0]B[0,4] + A[1,1]B[1,4]
    
    # Term 1: A[0,0] * B[0,4] -> C[0,0]
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 2: A[0,1] * B[1,4] -> C[0,0]
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 3: A[1,0] * B[0,4] -> C[1,0]
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Term 4: A[1,1] * B[1,4] -> C[1,0]
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return {"U": U, "V": V, "W": W}
