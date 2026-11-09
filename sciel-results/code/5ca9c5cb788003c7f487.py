"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen decomposition for a 2x2 block
    # x_idx: indices in U for x11, x12, x21, x22
    # y_idx: indices in V for y11, y12, y21, y22
    # z_idx: indices in W for z11, z12, z21, z22
    # base_col: starting column index
    def add_strassen(x_idx, y_idx, z_idx, base_col):
        x11, x12, x21, x22 = x_idx
        y11, y12, y21, y22 = y_idx
        z11, z12, z21, z22 = z_idx
        
        # M1 = (x11 + x22)(y11 + y22)
        U[x11][base_col] += 1
        U[x22][base_col] += 1
        V[y11][base_col] += 1
        V[y22][base_col] += 1
        W[z11][base_col] += 1
        W[z22][base_col] += 1
        
        # M2 = (x21 + x22)y11
        U[x21][base_col+1] += 1
        U[x22][base_col+1] += 1
        V[y11][base_col+1] += 1
        W[z21][base_col+1] += 1
        W[z22][base_col+1] -= 1
        
        # M3 = x11(y12 - y22)
        U[x11][base_col+2] += 1
        V[y12][base_col+2] += 1
        V[y22][base_col+2] -= 1
        W[z12][base_col+2] += 1
        W[z22][base_col+2] += 1
        
        # M4 = x22(y21 - y11)
        U[x22][base_col+3] += 1
        V[y21][base_col+3] += 1
        V[y11][base_col+3] -= 1
        W[z11][base_col+3] -= 1
        W[z21][base_col+3] += 1
        
        # M5 = (x11 + x12)y22
        U[x11][base_col+4] += 1
        U[x12][base_col+4] += 1
        V[y22][base_col+4] += 1
        W[z11][base_col+4] += 1
        W[z12][base_col+4] += 1
        
        # M6 = (x21 - x11)(y11 + y12)
        U[x21][base_col+5] += 1
        U[x11][base_col+5] -= 1
        V[y11][base_col+5] += 1
        V[y12][base_col+5] += 1
        W[z22][base_col+5] += 1
        
        # M7 = (x12 - x22)(y21 + y22)
        U[x12][base_col+6] += 1
        U[x22][base_col+6] -= 1
        V[y21][base_col+6] += 1
        V[y22][base_col+6] += 1
        W[z11][base_col+6] -= 1

    # Loop over the 3 column blocks of C (each 2x2)
    for c in range(3):
        base_col = c * 14
        
        # Block 1: A_left (2x2) * B_top (2x2) -> C_top (2x2)
        # A_left indices in U (A is 2x4, row-major)
        # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        x_idx_left = (0, 1, 2, 3)
        
        # B_top indices in V (B is 4x6, row-major)
        # B[0, 2c], B[0, 2c+1], B[1, 2c], B[1, 2c+1]
        # V index = i * 6 + j
        y_idx_top = (0 * 6 + 2*c, 0 * 6 + 2*c + 1, 1 * 6 + 2*c, 1 * 6 + 2*c + 1)
        
        # C_top indices in W (C is 2x6, W index = k * 2 + i)
        # C[0, 2c], C[0, 2c+1], C[1, 2c], C[1, 2c+1]
        # W index = col * 2 + row
        z_idx_top = (2*c * 2 + 0, 2*c * 2 + 1, 2*c * 2 + 2, 2*c * 2 + 3)
        
        add_strassen(x_idx_left, y_idx_top, z_idx_top, base_col)
        
        # Block 2: A_right (2x2) * B_bottom (2x2) -> C_bottom (2x2)
        # A_right indices in U
        # A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
        x_idx_right = (4, 5, 6, 7)
        
        # B_bottom indices in V
        # B[2, 2c], B[2, 2c+1], B[3, 2c], B[3, 2c+1]
        y_idx_bottom = (2 * 6 + 2*c, 2 * 6 + 2*c + 1, 3 * 6 + 2*c, 3 * 6 + 2*c + 1)
        
        # C_bottom indices in W
        # Same as C_top because it contributes to the same C block?
        # Yes, C = A_left B_top + A_right B_bottom.
        # So the output Z is the same 2x2 block of C.
        z_idx_bottom = z_idx_top
        
        add_strassen(x_idx_right, y_idx_bottom, z_idx_bottom, base_col + 7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
