import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The minimal rank for 2x4x4 matrix multiplication is 28.
    # This is achieved by applying Strassen's algorithm (rank 7) to four 2x2x2 block multiplications.
    # A is 2x4, B is 4x4, C is 2x4.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [[B_LL, B_LR], [B_RL, B_RR]] where blocks are 2x2.
    # C_L = A_L B_LL + A_R B_RL
    # C_R = A_L B_LR + A_R B_RR
    
    # We define the 4 Strassen instances.
    # Instance 1: A_L * B_LL -> contributes to C_L
    # Instance 2: A_R * B_RL -> contributes to C_L
    # Instance 3: A_L * B_LR -> contributes to C_R
    # Instance 4: A_R * B_RR -> contributes to C_R
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 terms
    # Indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    def add_strassen_block(u_offset_row, u_offset_col, 
                           v_offset_row, v_offset_col, 
                           w_offset_col, 
                           start_col):
        """
        Adds 7 Strassen terms for a 2x2x2 multiplication.
        
        u_offset_row: row offset for A block (0 or 1, but A is 2x4 so rows are 0,1. Wait, A is 2 rows. 
                      The 'row' offset in the 2x2 block sense refers to the local row 0 or 1.
                      But A is fixed to rows 0 and 1. The 'block' is defined by columns.
                      So u_offset_col is 0 (cols 0,1) or 1 (cols 2,3).
                      u_offset_row is always 0? No, the local row index r in 0..1 maps to global row r.
                      
        v_offset_row: row offset for B block (0 for rows 0,1; 1 for rows 2,3).
        v_offset_col: col offset for B block (0 for cols 0,1; 1 for cols 2,3).
        
        w_offset_col: col offset for C block (0 for cols 0,1; 1 for cols 2,3).
                      w_offset_row is always 0? Local row r maps to global row r.
        """
        
        # Strassen coefficients
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Output P:
        # P00 = M1 + M4 - M5 + M7
        # P01 = M3 + M5
        # P10 = M2 + M4
        # P11 = M1 - M2 + M3 + M6
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # M1
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            # M2
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            # M3
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            # M4
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            # M5
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            # M6
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            # M7
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        ]
        
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(terms):
            col = start_col + t_idx
            
            # Map U (A)
            # Local (r, c) -> Global (r, u_offset_col*2 + c)
            # Index in U: global_row * 4 + global_col
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = u_offset_col * 2 + c
                g_idx = g_r * 4 + g_c
                U[g_idx, col] += coeff
                
            # Map V (B)
            # Local (r, c) -> Global (v_offset_row*2 + r, v_offset_col*2 + c)
            # Index in V: global_row * 4 + global_col
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = v_offset_row * 2 + r
                g_c = v_offset_col * 2 + c
                g_idx = g_r * 4 + g_c
                V[g_idx, col] += coeff
                
            # Map W (C)
            # Local (r, c) -> Global (r, w_offset_col*2 + c)
            # Index in W: global_col * 2 + global_row (Trace convention: W[k*n+i])
            # k is col, i is row.
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = w_offset_col * 2 + c
                g_idx = g_c * 2 + g_r
                W[g_idx, col] += coeff

    # Apply the 4 blocks
    # 1. A_L (cols 0,1) * B_LL (rows 0,1; cols 0,1) -> C_L (cols 0,1)
    add_strassen_block(0, 0, 0, 0, 0)
    
    # 2. A_R (cols 2,3) * B_RL (rows 2,3; cols 0,1) -> C_L (cols 0,1)
    add_strassen_block(1, 1, 0, 0, 7)
    
    # 3. A_L (cols 0,1) * B_LR (rows 0,1; cols 2,3) -> C_R (cols 2,3)
    add_strassen_block(0, 0, 1, 1, 14)
    
    # 4. A_R (cols 2,3) * B_RR (rows 2,3; cols 2,3) -> C_R (cols 2,3)
    add_strassen_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
