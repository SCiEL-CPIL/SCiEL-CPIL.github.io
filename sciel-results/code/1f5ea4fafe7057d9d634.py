"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a structured rank-28 decomposition based on Strassen's algorithm
applied to 2x2 blocks.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix using Strassen's algorithm on 2x2 blocks.
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Define Strassen factors for 2x2x2
    # Local indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    strassen_factors = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]), # M1
        ([2, 3], [1, 1], [0], [1], [0, 2], [-1, 1]),       # M2
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),       # M3
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),       # M4
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),       # M5
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),       # M6
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),       # M7
    ]
    
    # Define blocks: (A_col_offset, B_row_offset, B_col_offset)
    # A is 2x4, B is 4x4, C is 2x4
    # Block 1: A[:, 0:2], B[0:2, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 0:2], B[0:2, 2:4] -> C[:, 2:4]
    # Block 3: A[:, 2:4], B[2:4, 0:2] -> C[:, 0:2]
    # Block 4: A[:, 2:4], B[2:4, 2:4] -> C[:, 2:4]
    
    blocks = [
        (0, 0, 0), # A_col_off, B_row_off, B_col_off
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2),
    ]
    
    col = 0
    for a_off, b_row_off, b_col_off in blocks:
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in strassen_factors:
            # Map U indices
            # Local A idx 0,1,2,3 -> Global A idx
            # Local 0: (0,0) -> Global (0, a_off+0) -> 0*4 + a_off
            # Local 1: (0,1) -> Global (0, a_off+1) -> 0*4 + a_off+1
            # Local 2: (1,0) -> Global (1, a_off+0) -> 1*4 + a_off
            # Local 3: (1,1) -> Global (1, a_off+1) -> 1*4 + a_off+1
            
            for li, val in zip(u_idx, u_val):
                if li == 0: g_idx = 0 * 4 + a_off
                elif li == 1: g_idx = 0 * 4 + a_off + 1
                elif li == 2: g_idx = 1 * 4 + a_off
                elif li == 3: g_idx = 1 * 4 + a_off + 1
                U[g_idx][col] += val
                
            # Map V indices
            # Local B idx 0,1,2,3 -> Global B idx
            # Local 0: (0,0) -> Global (b_row_off+0, b_col_off+0) -> (b_row_off)*4 + b_col_off
            # Local 1: (0,1) -> Global (b_row_off+0, b_col_off+1) -> (b_row_off)*4 + b_col_off+1
            # Local 2: (1,0) -> Global (b_row_off+1, b_col_off+0) -> (b_row_off+1)*4 + b_col_off
            # Local 3: (1,1) -> Global (b_row_off+1, b_col_off+1) -> (b_row_off+1)*4 + b_col_off+1
            
            for li, val in zip(v_idx, v_val):
                if li == 0: g_idx = b_row_off * 4 + b_col_off
                elif li == 1: g_idx = b_row_off * 4 + b_col_off + 1
                elif li == 2: g_idx = (b_row_off + 1) * 4 + b_col_off
                elif li == 3: g_idx = (b_row_off + 1) * 4 + b_col_off + 1
                V[g_idx][col] += val
                
            # Map W indices
            # Local C idx 0,1,2,3 -> Global C idx
            # Local 0: (0,0) -> Global (0, b_col_off+0) -> (b_col_off)*2 + 0
            # Local 1: (0,1) -> Global (0, b_col_off+1) -> (b_col_off+1)*2 + 0
            # Local 2: (1,0) -> Global (1, b_col_off+0) -> (b_col_off)*2 + 1
            # Local 3: (1,1) -> Global (1, b_col_off+1) -> (b_col_off+1)*2 + 1
            
            for li, val in zip(w_idx, w_val):
                if li == 0: g_idx = b_col_off * 2 + 0
                elif li == 1: g_idx = (b_col_off + 1) * 2 + 0
                elif li == 2: g_idx = b_col_off * 2 + 1
                elif li == 3: g_idx = (b_col_off + 1) * 2 + 1
                W[g_idx][col] += val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
