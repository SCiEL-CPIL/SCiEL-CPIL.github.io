"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct 3x3 schoolbook decomposition (Rank 27)
    # U_B, V_B, W_B shapes (9, 27)
    # Indexing: U_B[I*3+J, c], V_B[J*3+K, c], W_B[K*3+I, c]
    U_B = [[0] * 27 for _ in range(9)]
    V_B = [[0] * 27 for _ in range(9)]
    W_B = [[0] * 27 for _ in range(9)]
    c = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                U_B[I * 3 + J][c] = 1
                V_B[J * 3 + K][c] = 1
                W_B[K * 3 + I][c] = 1
                c += 1
                
    # 2. Construct 2x2 Strassen decomposition (Rank 7)
    # U_S, V_S, W_S shapes (4, 7)
    # Indexing: U_S[i'*2+j', c], V_S[j'*2+k', c], W_S[k'*2+i', c]
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U_S = [
        [1, 0, 1, 0, 1, -1, 0],  # i'=0, j'=0 (A11)
        [0, 0, 0, 0, 1, 0, 1],   # i'=0, j'=1 (A12)
        [0, 1, 0, 0, 0, 1, 0],   # i'=1, j'=0 (A21)
        [1, 1, 0, 1, 0, 0, -1]   # i'=1, j'=1 (A22)
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0],  # j'=0, k'=0 (B11)
        [0, 0, 1, 0, 0, 1, 0],   # j'=0, k'=1 (B12)
        [0, 0, 0, 1, 0, 0, 1],   # j'=1, k'=0 (B21)
        [1, 0, -1, 0, 1, 0, 1]   # j'=1, k'=1 (B22)
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1],  # k'=0, i'=0 (C11)
        [0, 0, 1, 0, 1, 0, 0],   # k'=0, i'=1 (C12)
        [0, 1, 0, 1, 0, 0, 0],   # k'=1, i'=0 (C21)
        [1, -1, 1, 0, 0, 1, 0]   # k'=1, i'=1 (C22)
    ]
    
    # 3. Combine to 6x6 decomposition
    # Rank = 27 * 7 = 189
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for c_B in range(27):
        for c_S in range(7):
            col = c_B * 7 + c_S
            
            # Fill U
            for I in range(3):
                for J in range(3):
                    u_B = U_B[I * 3 + J][c_B]
                    if u_B != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_S = U_S[i_prime * 2 + j_prime][c_S]
                                if u_S != 0:
                                    # Row index in 6x6: (2*I + i_prime)*6 + (2*J + j_prime)
                                    row = (2 * I + i_prime) * 6 + (2 * J + j_prime)
                                    U[row][col] = u_B * u_S
                                    
            # Fill V
            for J in range(3):
                for K in range(3):
                    v_B = V_B[J * 3 + K][c_B]
                    if v_B != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_S = V_S[j_prime * 2 + k_prime][c_S]
                                if v_S != 0:
                                    # Row index in 6x6: (2*J + j_prime)*6 + (2*K + k_prime)
                                    row = (2 * J + j_prime) * 6 + (2 * K + k_prime)
                                    V[row][col] = v_B * v_S
                                    
            # Fill W
            for K in range(3):
                for I in range(3):
                    w_B = W_B[K * 3 + I][c_B]
                    if w_B != 0:
                        for k_prime in range(2):
                            for i_prime in range(2):
                                w_S = W_S[k_prime * 2 + i_prime][c_S]
                                if w_S != 0:
                                    # Row index in 6x6: k*n + i = (2*K + k_prime)*6 + (2*I + i_prime)
                                    row = (2 * K + k_prime) * 6 + (2 * I + i_prime)
                                    W[row][col] = w_B * w_S
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
