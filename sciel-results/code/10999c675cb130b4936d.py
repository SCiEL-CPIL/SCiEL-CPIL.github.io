import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen on 4 blocks of 2x2x2
    # A is 2x4, split into A_L (2x2) and A_R (2x2)
    # B is 4x4, split into B11, B12 (top 2 rows) and B21, B22 (bottom 2 rows)
    # C is 2x4, C_L = A_L B11 + A_R B21, C_R = A_L B12 + A_R B22
    
    # We use 4 instances of Strassen (rank 7 each) -> Total 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # Inputs: X00, X01, X10, X11; Y00, Y01, Y10, Y11
    # Outputs: Z00, Z01, Z10, Z11
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X00+X11)Y10
    # M3 = X00(Y01-Y11)
    # M4 = X10(Y00+Y11)
    # M5 = (X01-X10)(Y10+Y11)
    # M6 = (X00-X10)(Y01+Y11)
    # M7 = (X11-X00)(Y00+Y10)
    
    # Z00 = M1 - M4 + M6 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M3 + M5 + M7
    
    # Indices for U (A):
    # A_L: 0, 1, 4, 5  (A00, A01, A10, A11)
    # A_R: 2, 3, 6, 7  (A02, A03, A12, A13)
    
    # Indices for V (B):
    # B11: 0, 1, 4, 5
    # B12: 2, 3, 6, 7
    # B21: 8, 9, 12, 13
    # B22: 10, 11, 14, 15
    
    # Indices for W (C):
    # C_L: 0, 1, 4, 5 (k=0,1; i=0,1 -> idx k*2+i)
    # C_R: 2, 3, 6, 7 (k=2,3; i=0,1 -> idx k*2+i)
    # Note: W index is k*n + i.
    # C_L (cols 0,1):
    # C00 (i=0, k=0) -> 0
    # C10 (i=1, k=0) -> 1
    # C01 (i=0, k=1) -> 2
    # C11 (i=1, k=1) -> 3
    # C_R (cols 2,3):
    # C02 (i=0, k=2) -> 4
    # C12 (i=1, k=2) -> 5
    # C03 (i=0, k=3) -> 6
    # C13 (i=1, k=3) -> 7

    def add_strassen_block(u_indices, v_indices, w_indices, col_offset):
        # u_indices: [X00, X01, X10, X11]
        # v_indices: [Y00, Y01, Y10, Y11]
        # w_indices: [Z00, Z01, Z10, Z11]
        
        # M1
        U[u_indices[0]][col_offset + 0] += 1
        U[u_indices[3]][col_offset + 0] += 1
        V[v_indices[0]][col_offset + 0] += 1
        V[v_indices[3]][col_offset + 0] += 1
        W[w_indices[0]][col_offset + 0] += 1
        W[w_indices[3]][col_offset + 0] += 1
        
        # M2
        U[u_indices[0]][col_offset + 1] += 1
        U[u_indices[3]][col_offset + 1] += 1
        V[v_indices[2]][col_offset + 1] += 1
        W[w_indices[2]][col_offset + 1] += 1
        
        # M3
        U[u_indices[0]][col_offset + 2] += 1
        V[v_indices[1]][col_offset + 2] += 1
        V[v_indices[3]][col_offset + 2] -= 1
        W[w_indices[1]][col_offset + 2] += 1
        
        # M4
        U[u_indices[2]][col_offset + 3] += 1
        V[v_indices[0]][col_offset + 3] += 1
        V[v_indices[3]][col_offset + 3] += 1
        W[w_indices[0]][col_offset + 3] -= 1
        W[w_indices[2]][col_offset + 3] += 1
        
        # M5
        U[u_indices[1]][col_offset + 4] += 1
        U[u_indices[2]][col_offset + 4] -= 1
        V[v_indices[2]][col_offset + 4] += 1
        V[v_indices[3]][col_offset + 4] += 1
        W[w_indices[1]][col_offset + 4] += 1
        W[w_indices[3]][col_offset + 4] += 1
        
        # M6
        U[u_indices[0]][col_offset + 5] += 1
        U[u_indices[2]][col_offset + 5] -= 1
        V[v_indices[1]][col_offset + 5] += 1
        V[v_indices[3]][col_offset + 5] += 1
        W[w_indices[0]][col_offset + 5] += 1
        
        # M7
        U[u_indices[3]][col_offset + 6] += 1
        U[u_indices[0]][col_offset + 6] -= 1
        V[v_indices[0]][col_offset + 6] += 1
        V[v_indices[2]][col_offset + 6] += 1
        W[w_indices[0]][col_offset + 6] -= 1
        W[w_indices[3]][col_offset + 6] += 1

    # Block 1: A_L, B11 -> C_L part 1
    # A_L: 0, 1, 4, 5
    # B11: 0, 1, 4, 5
    # C_L: 0, 2, 1, 3 (Z00, Z01, Z10, Z11)
    # Wait, W indices:
    # Z00 is C00 -> 0
    # Z01 is C01 -> 2
    # Z10 is C10 -> 1
    # Z11 is C11 -> 3
    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)

    # Block 2: A_R, B21 -> C_L part 2
    # A_R: 2, 3, 6, 7
    # B21: 8, 9, 12, 13
    # C_L: 0, 2, 1, 3
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 7)

    # Block 3: A_L, B12 -> C_R part 1
    # A_L: 0, 1, 4, 5
    # B12: 2, 3, 6, 7
    # C_R: 4, 6, 5, 7
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7], 14)

    # Block 4: A_R, B22 -> C_R part 2
    # A_R: 2, 3, 6, 7
    # B22: 10, 11, 14, 15
    # C_R: 4, 6, 5, 7
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
