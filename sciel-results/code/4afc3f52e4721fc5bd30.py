import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # We have 8 blocks.
    # Block index b = 0..7
    # b = 2 * j_block + k_block? No.
    # j_block in {0, 1} (cols 0,1 or 2,3 of A)
    # k_block in {0, 1, 2, 3} (cols 0,1 or 2,3 ... of B)
    # Total 2 * 4 = 8 blocks.
    
    # Strassen coefficients for 2x2x2
    # U_A[s][i][j]
    # U_B[s][j][k]
    # U_C[s][i][k]
    
    # s=0: M1 = (A00+A11)(B00+B11)
    # s=1: M2 = (A10+A11)B00
    # s=2: M3 = A00(B01-B11)
    # s=3: M4 = A11B10
    # s=4: M5 = (A00+A01)B11
    # s=5: M6 = (A10-A00)(B00+B01)
    # s=6: M7 = (A01-A11)(B10+B11)
    
    # A coeffs (i, j)
    A_coeffs = [
        [[1, 0], [0, 1]], # s=0: A00, A11
        [[0, 0], [1, 1]], # s=1: A10, A11
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 1], [0, 0]], # s=4: A00, A01
        [[-1, 0], [1, 0]], # s=5: -A00, A10
        [[0, 1], [0, -1]]  # s=6: A01, -A11
    ]
    
    # B coeffs (j, k)
    B_coeffs = [
        [[1, 0], [0, 1]], # s=0: B00, B11
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 1], [0, -1]], # s=2: B01, -B11
        [[0, 0], [1, 0]], # s=3: B10
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 1], [0, 0]], # s=5: B00, B01
        [[0, 0], [1, 1]]  # s=6: B10, B11
    ]
    
    # C coeffs (i, k)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [[1, 0], [0, 1]], # s=0 (M1): C00, C11
        [[0, 0], [1, -1]], # s=1 (M2): C10, -C11
        [[0, 1], [0, 1]], # s=2 (M3): C01, C11
        [[1, 0], [1, 0]], # s=3 (M4): C00, C10
        [[-1, 1], [0, 0]], # s=4 (M5): -C00, C01
        [[0, 0], [0, 1]], # s=5 (M6): C11
        [[1, 0], [0, 0]]  # s=6 (M7): C00
    ]
    
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j_block in range(2):
        for k_block in range(4):
            for s in range(7):
                # j_block determines which 2 cols of A (and 2 rows of B)
                # j_global = j_block * 2 + j_local
                # k_block determines which 2 cols of B (and 2 cols of C)
                # k_global = k_block * 2 + k_local
                
                for i_local in range(2):
                    for j_local in range(2):
                        j_global = j_block * 2 + j_local
                        i_global = i_local # i is 0 or 1
                        
                        # U index: i * m + j
                        u_idx = i_global * m + j_global
                        U[u_idx][col] = A_coeffs[s][i_local][j_local]
                
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_block * 2 + j_local
                        k_global = k_block * 2 + k_local
                        
                        # V index: j * p + k
                        v_idx = j_global * p + k_global
                        V[v_idx][col] = B_coeffs[s][j_local][k_local]
                
                for i_local in range(2):
                    for k_local in range(2):
                        i_global = i_local
                        k_global = k_block * 2 + k_local
                        
                        # W index: k * n + i
                        w_idx = k_global * n + i_global
                        W[w_idx][col] = C_coeffs[s][i_local][k_local]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
