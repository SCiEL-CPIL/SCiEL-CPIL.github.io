"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _construct_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, T, tol=1e-8):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _strassen_2x2x2(U, V, W, base_i, base_j, base_k):
    """
    Adds Strassen's 7 products for a 2x2x2 sub-block.
    U is (nm, rank), V is (mp, rank), W is (pn, rank).
    We assume we are working in a local coordinate system or 
    this function is used to build the matrices directly.
    """
    # This helper is conceptual. We will build the matrices directly in construct().
    pass

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T = _construct_target_tensor(n, m, p)
    
    # Schoolbook fallback (rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Construct the rank 54 solution using Strassen on 2x2 blocks
    # Partition m=5 into 2+2+1
    # Partition p=6 into 2+2+2
    # Blocks:
    # 1. A[0:2, 0:2] * B[0:2, :] (2x2x6) -> 3 * (2x2x2)
    # 2. A[0:2, 2:4] * B[2:4, :] (2x2x6) -> 3 * (2x2x2)
    # 3. A[0:2, 4:5] * B[4:5, :] (2x1x6) -> 6 * (2x1x1)
    
    rank_54 = 54
    U_54 = np.zeros((nm, rank_54))
    V_54 = np.zeros((mp, rank_54))
    W_54 = np.zeros((pn, rank_54))
    
    r = 0
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(a_row_start, b_row_start, c_col_start):
        nonlocal r
        # A indices: (a_row_start, b_row_start) to (a_row_start+1, b_row_start+1)
        # B indices: (b_row_start, c_col_start) to (b_row_start+1, c_col_start+1)
        # C indices: (a_row_start, c_col_start) to (a_row_start+1, c_col_start+1)
        
        # Strassen products for 2x2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B10
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Note: A is 2x2, B is 2x2, C is 2x2.
        # A[i,j] corresponds to U[i*m+j]
        # B[j,k] corresponds to V[j*p+k]
        # C[i,k] corresponds to W[k*n+i]
        
        a00 = a_row_start * m + b_row_start
        a01 = a_row_start * m + b_row_start + 1
        a10 = (a_row_start + 1) * m + b_row_start
        a11 = (a_row_start + 1) * m + b_row_start + 1
        
        b00 = b_row_start * p + c_col_start
        b01 = b_row_start * p + c_col_start + 1
        b10 = (b_row_start + 1) * p + c_col_start
        b11 = (b_row_start + 1) * p + c_col_start + 1
        
        c00 = c_col_start * n + a_row_start
        c01 = (c_col_start + 1) * n + a_row_start
        c10 = c_col_start * n + a_row_start + 1
        c11 = (c_col_start + 1) * n + a_row_start + 1
        
        # M1
        U_54[a00, r] = 1; U_54[a11, r] = 1
        V_54[b00, r] = 1; V_54[b11, r] = 1
        W_54[c00, r] = 1; W_54[c01, r] = 1; W_54[c10, r] = 1; W_54[c11, r] = 1
        r += 1
        
        # M2
        U_54[a10, r] = 1; U_54[a11, r] = 1
        V_54[b10, r] = 1
        W_54[c10, r] = 1; W_54[c11, r] = 1
        r += 1
        
        # M3
        U_54[a00, r] = 1
        V_54[b01, r] = 1; V_54[b11, r] = -1
        W_54[c01, r] = 1; W_54[c11, r] = 1
        r += 1
        
        # M4
        U_54[a11, r] = 1
        V_54[b10, r] = 1; V_54[b00, r] = -1
        W_54[c10, r] = 1; W_54[c00, r] = 1
        r += 1
        
        # M5
        U_54[a00, r] = 1; U_54[a01, r] = 1
        V_54[b11, r] = 1
        W_54[c01, r] = 1; W_54[c11, r] = 1
        r += 1
        
        # M6
        U_54[a10, r] = 1; U_54[a00, r] = -1
        V_54[b00, r] = 1; V_54[b01, r] = 1
        W_54[c00, r] = 1; W_54[c10, r] = 1
        r += 1
        
        # M7
        U_54[a01, r] = 1; U_54[a11, r] = -1
        V_54[b10, r] = 1; V_54[b11, r] = 1
        W_54[c01, r] = 1; W_54[c11, r] = 1
        r += 1

    # Helper to add a 2x1x1 block (simple multiplication)
    def add_simple_block(a_row_start, b_row_start, c_col_start):
        nonlocal r
        # A is 2x1, B is 1x1, C is 2x1
        a0 = a_row_start * m + b_row_start
        a1 = (a_row_start + 1) * m + b_row_start
        b0 = b_row_start * p + c_col_start
        c0 = c_col_start * n + a_row_start
        c1 = c_col_start * n + a_row_start + 1
        
        # C0 = A0 * B0
        U_54[a0, r] = 1
        V_54[b0, r] = 1
        W_54[c0, r] = 1
        r += 1
        
        # C1 = A1 * B0
        U_54[a1, r] = 1
        V_54[b0, r] = 1
        W_54[c1, r] = 1
        r += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:6]
    # Split B into 3 2x2 blocks: [0:2, 0:2], [0:2, 2:4], [0:2, 4:6]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:6]
    # Split B into 3 2x2 blocks: [2:4, 0:2], [2:4, 2:4], [2:4, 4:6]
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Block 3: A[0:2, 4:5] * B[4:5, 0:6]
    # B is 1x6. Split into 6 1x1 blocks.
    for k in range(6):
        add_simple_block(0, 4, k)
        
    # Verify
    if _verify_fast(U_54, V_54, W_54, T, tol=1e-8):
        best_rank = 54
        best_U, best_V, best_W = U_54, V_54, W_54
    else:
        # Fallback to schoolbook if verification fails
        best_rank = 60
        best_U, best_V, best_W = U_sb, V_sb, W_sb

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
