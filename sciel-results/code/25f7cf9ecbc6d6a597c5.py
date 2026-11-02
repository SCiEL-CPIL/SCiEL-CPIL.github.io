"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 84 using Strassen decomposition on 2x2x2 blocks
    rank = 84
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # Each term is a list of (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # Local indices 0..1 for rows/cols of 2x2 blocks
    
    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        # Local to Global index mapping functions
        def u_idx(lr, lc):
            return (a_row_off + lr) * m + (a_col_off + lc)
        def v_idx(lr, lc):
            return (b_row_off + lr) * p + (b_col_off + lc)
        def w_idx(lr, lc):
            return (c_col_off + lc) * n + (c_row_off + lr)
            
        # 1. M1 = (X00+X11)(Y00+Y11) -> Z00 +1, Z11 +1
        U[u_idx(0,0), col] += 1; U[u_idx(1,1), col] += 1
        V[v_idx(0,0), col] += 1; V[v_idx(1,1), col] += 1
        W[w_idx(0,0), col] += 1; W[w_idx(1,1), col] += 1
        col += 1
        
        # 2. M2 = (X10+X11)Y00 -> Z10 +1, Z11 -1
        U[u_idx(1,0), col] += 1; U[u_idx(1,1), col] += 1
        V[v_idx(0,0), col] += 1
        W[w_idx(1,0), col] += 1; W[w_idx(1,1), col] -= 1
        col += 1
        
        # 3. M3 = X00(Y01-Y11) -> Z01 +1, Z11 +1
        U[u_idx(0,0), col] += 1
        V[v_idx(0,1), col] += 1; V[v_idx(1,1), col] -= 1
        W[w_idx(0,1), col] += 1; W[w_idx(1,1), col] += 1
        col += 1
        
        # 4. M4 = X11(Y10-Y00) -> Z00 +1, Z10 +1
        U[u_idx(1,1), col] += 1
        V[v_idx(1,0), col] += 1; V[v_idx(0,0), col] -= 1
        W[w_idx(0,0), col] += 1; W[w_idx(1,0), col] += 1
        col += 1
        
        # 5. M5 = (X00+X01)Y11 -> Z00 -1, Z01 +1
        U[u_idx(0,0), col] += 1; U[u_idx(0,1), col] += 1
        V[v_idx(1,1), col] += 1
        W[w_idx(0,0), col] -= 1; W[w_idx(0,1), col] += 1
        col += 1
        
        # 6. M6 = (X10-X00)(Y00+Y01) -> Z11 +1
        U[u_idx(1,0), col] += 1; U[u_idx(0,0), col] -= 1
        V[v_idx(0,0), col] += 1; V[v_idx(0,1), col] += 1
        W[w_idx(1,1), col] += 1
        col += 1
        
        # 7. M7 = (X01-X11)(Y10+Y11) -> Z00 +1
        U[u_idx(0,1), col] += 1; U[u_idx(1,1), col] -= 1
        V[v_idx(1,0), col] += 1; V[v_idx(1,1), col] += 1
        W[w_idx(0,0), col] += 1
        col += 1

    # 12 blocks of 2x2x2
    # A blocks: (0,0), (0,1), (1,0), (1,1) in 2x2 grid of 2x2 blocks
    # B blocks: Top (rows 0-1) and Bottom (rows 2-3)
    # B cols: 0-1, 2-3, 4-5
    
    # A00 (rows 0-1, cols 0-1)
    add_strassen_block(0, 0, 0, 0, 0, 0) # Bt0
    add_strassen_block(0, 0, 0, 2, 0, 2) # Bt1
    add_strassen_block(0, 0, 0, 4, 0, 4) # Bt2
    
    # A01 (rows 0-1, cols 2-3)
    add_strassen_block(0, 2, 2, 0, 0, 0) # Bb0
    add_strassen_block(0, 2, 2, 2, 0, 2) # Bb1
    add_strassen_block(0, 2, 2, 4, 0, 4) # Bb2
    
    # A10 (rows 2-3, cols 0-1)
    add_strassen_block(2, 0, 0, 0, 2, 0) # Bt0
    add_strassen_block(2, 0, 0, 2, 2, 2) # Bt1
    add_strassen_block(2, 0, 0, 4, 2, 4) # Bt2
    
    # A11 (rows 2-3, cols 2-3)
    add_strassen_block(2, 2, 2, 0, 2, 0) # Bb0
    add_strassen_block(2, 2, 2, 2, 2, 2) # Bb1
    add_strassen_block(2, 2, 2, 4, 2, 4) # Bb2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
