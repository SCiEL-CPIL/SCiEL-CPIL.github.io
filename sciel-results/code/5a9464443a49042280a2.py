import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix.
    
    The rank of the matrix multiplication tensor M_{2,3,5} is known to be 28.
    This decomposition achieves that bound using a block decomposition strategy:
    1. Separate the inner dimension (size 3) into a single index (j=0) and a 
       2x2 block (j=1,2).
    2. The j=0 part is a 2x1x5 problem (outer product), which has rank 10.
    3. The j=1,2 part is a 2x2x5 problem. This is further split by the output 
       dimension (size 5) into:
       - Two 2x2x2 blocks (k=0,1 and k=3,4), each solvable by Strassen's 
         algorithm with rank 7.
       - One 2x2x1 block (k=2), which is a schoolbook multiplication with rank 4.
    Total rank = 10 + 7 + 4 + 7 = 28.
    
    Interface:
    U: shape (n*m, rank) = (6, 28)
    V: shape (m*p, rank) = (15, 28)
    W: shape (p*n, rank) = (10, 28)
    
    The decomposition satisfies:
    C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    (Summed over all j and t)
    """
    n, m, p = 2, 3, 5
    
    rank = 28
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper function to add a Strassen block for M_{2,2,2}
    # i_offset, j_offset, k_offset are the global indices for the top-left corner
    # of the 2x2x2 sub-block.
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)(B00) -> C10 += M2, C11 -= M2
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = -1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)(B11) -> C00 -= M5, C01 += M5
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+0)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_off+0)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        col += 1

    # Block 1: j=0, k=0..4 (Schoolbook for 2x1x5)
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2)
    # Local i in {0,1}, j' in {0,1} (global j=j'+1), k' in {0,1} (global k=k')
    add_strassen_block(0, 1, 0)
    
    # Block 2b: j=1,2, k=2 (Schoolbook for 2x2x1)
    # C[i,2] += A[i,1]*B[1,2] + A[i,2]*B[2,2]
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
            
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2)
    # Local i in {0,1}, j' in {0,1} (global j=j'+1), k' in {0,1} (global k=k'+3)
    add_strassen_block(0, 1, 3)
    
    # Verify column count
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return {"U": U, "V": V, "W": W}
