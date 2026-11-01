"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """
    Builds the rank-28 Strassen decomposition for 2x4 * 4x4 matrix multiplication.
    
    The matrix multiplication tensor T is defined such that:
    C[i, j] = sum_k A[i, k] B[k, j]
    T[i, k, j] = 1 for all i, k, j? No.
    The tensor entry T_{i,k,j} is 1 if the term A_{i,k} B_{k,j} contributes to C_{i,j}.
    In the standard matmul tensor, T_{i,k,j} = 1 for all valid indices.
    The decomposition is:
    T[i, k, j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    Here n=2, m=4, p=4.
    U has shape (2*4, 28) = (8, 28).
    V has shape (4*4, 28) = (16, 28).
    W has shape (4*2, 28) = (8, 28).
    
    We decompose the problem into 4 blocks of 2x2 * 2x2 -> 2x2.
    A is 2x4, split into A1 (cols 0,1) and A2 (cols 2,3).
    B is 4x4, split into B1 (rows 0,1), B2 (rows 2,3).
    B1 is 2x4, split into B1_L (cols 0,1) and B1_R (cols 2,3).
    B2 is 2x4, split into B2_L (cols 0,1) and B2_R (cols 2,3).
    
    C is 2x4.
    C_L (cols 0,1) = A1 * B1_L + A2 * B2_L
    C_R (cols 2,3) = A1 * B1_R + A2 * B2_R
    
    Each term like A1 * B1_L is a 2x2 * 2x2 multiplication.
    Strassen uses 7 multiplications for each.
    There are 4 such blocks:
    1. A1 * B1_L -> contributes to C_L
    2. A2 * B2_L -> contributes to C_L
    3. A1 * B1_R -> contributes to C_R
    4. A2 * B2_R -> contributes to C_R
    
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 7 Strassen terms for a generic 2x2 * 2x2 -> 2x2 multiplication.
    # Indices for local 2x2 matrices:
    # X (A block): 00, 01, 10, 11 -> flat indices 0, 1, 2, 3
    # Y (B block): 00, 01, 10, 11 -> flat indices 0, 1, 2, 3
    # P (C block): 00, 01, 10, 11 -> flat indices 0, 1, 2, 3
    
    # Strassen formulas:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    
    # Reconstructed P:
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # We define the coefficients for U, V, W for each of the 7 terms.
    # U coeffs (for X):
    U_terms = [
        {0: 1, 3: 1},       # M1: X00, X11
        {2: 1, 3: 1},       # M2: X10, X11
        {0: 1},             # M3: X00
        {3: 1},             # M4: X11
        {0: 1, 1: 1},       # M5: X00, X01
        {2: 1, 0: -1},      # M6: X10, -X00
        {1: 1, 3: -1},      # M7: X01, -X11
    ]
    
    # V coeffs (for Y):
    V_terms = [
        {0: 1, 3: 1},       # M1: Y00, Y11
        {0: 1},             # M2: Y00
        {1: 1, 3: -1},      # M3: Y01, -Y11
        {2: 1, 0: -1},      # M4: Y10, -Y00
        {3: 1},             # M5: Y11
        {0: 1, 1: 1},       # M6: Y00, Y01
        {2: 1, 3: 1},       # M7: Y10, Y11
    ]
    
    # W coeffs (for P):
    W_terms = [
        {0: 1, 3: 1},       # M1: P00, P11
        {2: 1, 3: -1},      # M2: P10, -P11
        {1: 1, 3: 1},       # M3: P01, P11
        {0: 1, 2: 1},       # M4: P00, P10
        {0: -1, 1: 1},      # M5: -P00, P01
        {3: 1},             # M6: P11
        {0: 1},             # M7: P00
    ]
    
    # Mapping of global indices
    # A (2x4): Row i (0,1), Col k (0,1,2,3). Index = i*4 + k.
    # B (4x4): Row k (0,1,2,3), Col j (0,1,2,3). Index = k*4 + j.
    # C (2x4): Row i (0,1), Col j (0,1,2,3). Index = j*2 + i (Trace convention W[k*n+i, r] where k is col, i is row).
    # Wait, the prompt says "W[k*n+i,r]". 
    # In the tensor decomposition T[i,k,j] = sum U[i*m+k, r] V[k*p+j, r] W[j*n+i, r].
    # Here the output index is (i, j). 
    # The W index in the formula is `j*n + i`.
    # j is the column of C (0..p-1). i is the row of C (0..n-1).
    # So W index = j*2 + i.
    
    # We have 4 blocks to process.
    # Block 0: A1 (cols 0,1) * B1_L (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # Block 1: A2 (cols 2,3) * B2_L (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # Block 2: A1 (cols 0,1) * B1_R (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # Block 3: A2 (cols 2,3) * B2_R (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    
    # Parameters for each block:
    # a_col_offset: 0 for A1, 1 for A2 (multiplied by 2)
    # b_row_offset: 0 for B1, 1 for B2 (multiplied by 2)
    # b_col_offset: 0 for L, 1 for R (multiplied by 2)
    # c_col_offset: 0 for L, 1 for R (multiplied by 2)
    
    blocks = [
        (0, 0, 0, 0), # Block 0
        (1, 1, 0, 0), # Block 1
        (0, 0, 1, 1), # Block 2
        (1, 1, 1, 1), # Block 3
    ]
    
    col_idx = 0
    for (a_off, b_row_off, b_col_off, c_col_off) in blocks:
        for s in range(7):
            # U contributions
            for loc_idx, coeff in U_terms[s].items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_off * 2 + c
                U[g_r * 4 + g_c, col_idx] += coeff
            
            # V contributions
            for loc_idx, coeff in V_terms[s].items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_off * 2 + r
                g_c = b_col_off * 2 + c
                V[g_r * 4 + g_c, col_idx] += coeff
            
            # W contributions
            for loc_idx, coeff in W_terms[s].items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_off * 2 + c
                # W index is g_c * n + g_r = g_c * 2 + g_r
                W[g_c * 2 + g_r, col_idx] += coeff
            
            col_idx += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify if U, V, W correctly decompose the matmul tensor.
    T[i, k, j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    For matmul, T[i, k, j] should be 1.
    """
    rank = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Build the rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should not happen, but handle it gracefully
        raise ValueError("Rank 28 decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
