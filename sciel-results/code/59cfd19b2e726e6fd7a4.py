"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-54 exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    
    Strategy:
    - Split the 5 middle columns into: 1 + 2 + 2 (i.e., col 0, cols 1-2, cols 3-4)
    - For each block (n x m_block x p):
      - 2x1x6: schoolbook rank = 1*2*6 = 12
      - 2x2x6: split into three 2x2x2 blocks, each using Strassen (rank 7), total = 3*7 = 21
    - Total rank = 12 + 21 + 21 = 54
    
    This is an improvement over the schoolbook rank of 60.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Start with all zeros; we'll fill in columns as we process each block.
    
    # Pre-allocate as lists of lists with 0s (we don't know final rank until we count, but we know it's 54)
    # Actually, let's just build up the matrices as lists of columns and then transpose at the end.
    # Or better: allocate with the known total rank of 54 and fill in.
    
    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 54
    
    col_idx = 0
    
    # Helper: add a 2x1x6 block (schoolbook)
    # A: rows i in [0,1], col j_off (1 column)
    # B: row j_off, cols k_off to k_off+5
    # C: rows k_off to k_off+5, cols i in [0,1]
    def add_block_2x1x6(j_off, k_off):
        nonlocal col_idx
        for i in range(2):
            for k in range(6):
                U[i * m + (j_off)][col_idx] = 1
                V[(j_off) * p + (k_off + k)][col_idx] = 1
                W[(k_off + k) * n + i][col_idx] = 1
                col_idx += 1
        # This adds 12 columns
    
    # Helper: add a 2x2x2 block using Strassen's algorithm (rank 7)
    # A: rows i in [0,1], cols j_off, j_off+1
    # B: rows j_off, j_off+1, cols k_off, k_off+1
    # C: rows k_off, k_off+1, cols i in [0,1]
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    def add_block_2x2x2(j_off, k_off):
        nonlocal col_idx
        # A row indices: (0, j_off), (0, j_off+1), (1, j_off), (1, j_off+1)
        # B row indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C row indices: (k_off, 0), (k_off, 1), (k_off+1, 0), (k_off+1, 1)
        
        a11 = 0 * m + j_off        # A[0][j_off]
        a12 = 0 * m + (j_off + 1)  # A[0][j_off+1]
        a21 = 1 * m + j_off        # A[1][j_off]
        a22 = 1 * m + (j_off + 1)  # A[1][j_off+1]
        
        b11 = j_off * p + k_off
        b12 = j_off * p + (k_off + 1)
        b21 = (j_off + 1) * p + k_off
        b22 = (j_off + 1) * p + (k_off + 1)
        
        c11 = k_off * n + 0
        c12 = k_off * n + 1
        c21 = (k_off + 1) * n + 0
        c22 = (k_off + 1) * n + 1
        
        # M1 = (A11+A22)(B11+B22)
        U[a11][col_idx] = 1
        U[a22][col_idx] = 1
        V[b11][col_idx] = 1
        V[b22][col_idx] = 1
        W[c11][col_idx] = 1
        W[c12][col_idx] = 1
        W[c21][col_idx] = 1
        W[c22][col_idx] = -1
        col_idx += 1
        
        # M2 = (A21+A22)B11
        U[a21][col_idx] = 1
        U[a22][col_idx] = 1
        V[b11][col_idx] = 1
        W[c21][col_idx] = 1
        W[c22][col_idx] = -1
        col_idx += 1
        
        # M3 = A11(B21-B22)
        U[a11][col_idx] = 1
        V[b21][col_idx] = 1
        V[b22][col_idx] = -1
        W[c12][col_idx] = 1
        W[c22][col_idx] = 1
        col_idx += 1
        
        # M4 = A22(B12-B11)
        U[a22][col_idx] = 1
        V[b12][col_idx] = 1
        V[b11][col_idx] = -1
        W[c11][col_idx] = 1
        W[c21][col_idx] = 1
        col_idx += 1
        
        # M5 = (A11+A12)B22
        U[a11][col_idx] = 1
        U[a12][col_idx] = 1
        V[b22][col_idx] = 1
        W[c11][col_idx] = -1
        W[c12][col_idx] = 1
        col_idx += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a21][col_idx] = 1
        U[a11][col_idx] = -1
        V[b11][col_idx] = 1
        V[b12][col_idx] = 1
        W[c22][col_idx] = 1
        col_idx += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a12][col_idx] = 1
        U[a22][col_idx] = -1
        V[b21][col_idx] = 1
        V[b22][col_idx] = 1
        W[c11][col_idx] = 1
        W[c22][col_idx] = 1
        col_idx += 1
        
        # This adds 7 columns
    
    # Helper: add a 2x2x6 block using three Strassen 2x2x2 blocks
    def add_block_2x2x6(j_off, k_off):
        nonlocal col_idx
        # Split the 6 columns into three groups of 2
        for kk in range(0, 6, 2):
            add_block_2x2x2(j_off, k_off + kk)
        # This adds 3 * 7 = 21 columns
    
    # Now process the blocks:
    # Block 1: 2x1x6 at j_off=0, k_off=0
    add_block_2x1x6(0, 0)
    
    # Block 2: 2x2x6 at j_off=1, k_off=0
    add_block_2x2x6(1, 0)
    
    # Block 3: 2x2x6 at j_off=3, k_off=0
    add_block_2x2x6(3, 0)
    
    # Verify col_idx == 54
    assert col_idx == 54, f"Expected 54 columns, got {col_idx}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
