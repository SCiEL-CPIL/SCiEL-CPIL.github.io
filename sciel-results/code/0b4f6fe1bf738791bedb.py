"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def block_4x4x2():
        U_local = np.zeros((16, 28))
        V_local = np.zeros((8, 28))
        W_local = np.zeros((8, 28))
        
        sU, sV, sW = strassen_2x2x2()
        
        col_offset = 0
        for r in range(7):
            for ai in range(4):
                U_local[ai, col_offset + r] = sU[ai, r]
            for bi in range(4):
                V_local[bi, col_offset + r] = sV[bi, r]
            for ci in range(4):
                k = ci // 2
                i = ci % 2
                W_local[k * 4 + i, col_offset + r] = sW[ci, r]
        
        col_offset = 7
        for r in range(7):
            for ai in range(4):
                U_local[4 + ai, col_offset + r] = sU[ai, r]
            for bi in range(4):
                V_local[4 + bi, col_offset + r] = sV[bi, r]
            for ci in range(4):
                k = ci // 2
                i = ci % 2
                W_local[k * 4 + i, col_offset + r] = sW[ci, r]
        
        col_offset = 14
        for r in range(7):
            for ai in range(4):
                U_local[8 + ai, col_offset + r] = sU[ai, r]
            for bi in range(4):
                V_local[bi, col_offset + r] = sV[bi, r]
            for ci in range(4):
                k = ci // 2
                i = ci % 2
                W_local[4 + k * 4 + i, col_offset + r] = sW[ci, r]
        
        col_offset = 21
        for r in range(7):
            for ai in range(4):
                U_local[12 + ai, col_offset + r] = sU[ai, r]
            for bi in range(4):
                V_local[4 + bi, col_offset + r] = sV[bi, r]
            for ci in range(4):
                k = ci // 2
                i = ci % 2
                W_local[4 + k * 4 + i, col_offset + r] = sW[ci, r]
        
        return U_local, V_local, W_local
    
    U_full = np.zeros((16, 84))
    V_full = np.zeros((24, 84))
    W_full = np.zeros((24, 84))
    
    for block in range(3):
        U_b, V_b, W_b = block_4x4x2()
        col_offset = block * 28
        U_full[:, col_offset:col_offset+28] = U_b
        for bi in range(8):
            b_row = bi // 2
            b_col = bi % 2
            B_idx = b_row * 6 + block * 2 + b_col
            V_full[B_idx, col_offset:col_offset+28] = V_b[bi, :]
        for ci in range(8):
            c_col_local = ci // 2
            c_row = ci % 2
            k = block * 2 + c_col_local
            i = c_row
            W_idx = k * 4 + i
            W_full[W_idx, col_offset:col_offset+28] = W_b[ci, :]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
