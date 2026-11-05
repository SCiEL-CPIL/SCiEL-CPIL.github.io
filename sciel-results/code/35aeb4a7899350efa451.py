"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 117
    # 8 Strassen blocks (rank 7 each) + 61 schoolbook terms (rank 1 each)
    # Blocks are on indices {0,1,2,3} for I, J, K
    # Remaining terms involve index 4
    
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_col = 0
    
    # Define the 8 blocks
    # I pairs: (0,1), (2,3)
    # J pairs: (0,1), (2,3)
    # K pairs: (0,1), (2,3)
    i_pairs = [(0, 1), (2, 3)]
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3)]
    
    # Strassen coefficients
    # M1: (A11+A22)(B11+B22) -> C11(1), C22(1)
    # M2: (A21+A22)B11 -> C21(1), C22(-1)
    # M3: A11(B12-B22) -> C12(1), C22(1)
    # M4: A22(B21-B11) -> C11(1), C21(1)
    # M5: (A11+A12)B22 -> C11(-1), C12(1)
    # M6: (A21-A11)(B11+B12) -> C22(1)
    # M7: (A12-A22)(B21+B22) -> C11(1)
    
    strassen_data = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: (i1j1, i1j2, i2j1, i2j2)
        # V: (j1k1, j1k2, j2k1, j2k2)
        # W: (k1i1, k1i2, k2i1, k2i2)
        (
            [1, 0, 1, 0], # M1
            [1, 0, 0, 1],
            [1, 0, 0, 1]
        ),
        (
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0],
            [0, 0, 1, -1]
        ),
        (
            [1, 0, 0, 0], # M3
            [0, 1, 0, -1],
            [0, 1, 0, 1]
        ),
        (
            [0, 0, 0, 1], # M4
            [0, 0, 1, -1],
            [1, 0, 1, 0]
        ),
        (
            [1, 1, 0, 0], # M5
            [0, 0, 0, 1],
            [-1, 1, 0, 0]
        ),
        (
            [0, -1, 1, 0], # M6
            [1, 1, 0, 0],
            [0, 0, 0, 1]
        ),
        (
            [0, 1, 0, -1], # M7
            [0, 0, 1, 1],
            [1, 0, 0, 0]
        )
    ]
    
    for i1, i2 in i_pairs:
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                for u_c, v_c, w_c in strassen_data:
                    # U
                    U[i1 * m + j1][current_col] += u_c[0]
                    U[i1 * m + j2][current_col] += u_c[1]
                    U[i2 * m + j1][current_col] += u_c[2]
                    U[i2 * m + j2][current_col] += u_c[3]
                    
                    # V
                    V[j1 * p + k1][current_col] += v_c[0]
                    V[j1 * p + k2][current_col] += v_c[1]
                    V[j2 * p + k1][current_col] += v_c[2]
                    V[j2 * p + k2][current_col] += v_c[3]
                    
                    # W
                    W[k1 * n + i1][current_col] += w_c[0]
                    W[k1 * n + i2][current_col] += w_c[1]
                    W[k2 * n + i1][current_col] += w_c[2]
                    W[k2 * n + i2][current_col] += w_c[3]
                    
                    current_col += 1
                    
    # Schoolbook for remaining terms
    # Terms where i=4 or j=4 or k=4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if all indices are in {0,1,2,3}
                if i < 4 and j < 4 and k < 4:
                    continue
                U[i * m + j][current_col] = 1
                V[j * p + k][current_col] = 1
                W[k * n + i][current_col] = 1
                current_col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
