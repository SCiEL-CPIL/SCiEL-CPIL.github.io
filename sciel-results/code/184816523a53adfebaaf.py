"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189  # 7 * 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 blocks
    # c_s[I][J], d_s[J][K], alpha_s[I][K]
    # s is 1-based in formulas, 0-based in list
    
    # M1: A00+A11, B00+B11
    c_1 = [[1, 0], [0, 1]]
    d_1 = [[1, 0], [0, 1]]
    a_1 = [[1, 0], [0, 1]]
    
    # M2: A10+A11, B00
    c_2 = [[0, 0], [1, 1]]
    d_2 = [[1, 0], [0, 0]]
    a_2 = [[0, 0], [1, -1]]
    
    # M3: A00, B01-B11
    c_3 = [[1, 0], [0, 0]]
    d_3 = [[0, 1], [0, -1]]
    a_3 = [[0, 1], [0, 1]]
    
    # M4: A11, B10-B00
    c_4 = [[0, 0], [0, 1]]
    d_4 = [[-1, 0], [1, 0]]
    a_4 = [[1, 0], [1, 0]]
    
    # M5: A00+A01, B11
    c_5 = [[1, 1], [0, 0]]
    d_5 = [[0, 0], [0, 1]]
    a_5 = [[-1, 0], [0, 1]]
    
    # M6: A10-A00, B00+B01
    c_6 = [[-1, 0], [1, 0]]
    d_6 = [[1, 1], [0, 0]]
    a_6 = [[0, 0], [0, 1]]
    
    # M7: A01-A11, B10+B11
    c_7 = [[0, 1], [-1, 0]]
    d_7 = [[0, 0], [1, 1]]
    a_7 = [[1, 0], [0, 0]]
    
    coeffs = [
        (c_1, d_1, a_1),
        (c_2, d_2, a_2),
        (c_3, d_3, a_3),
        (c_4, d_4, a_4),
        (c_5, d_5, a_5),
        (c_6, d_6, a_6),
        (c_7, d_7, a_7)
    ]
    
    for s_idx in range(7):
        c_s, d_s, a_s = coeffs[s_idx]
        for j_local in range(3):
            r = s_idx * 27 + j_local
            
            # Fill U
            # U[i*6+j, r] = c_s[I][J] if j%3 == j_local
            for I in range(2):
                for i_local in range(3):
                    i = I * 3 + i_local
                    for J in range(2):
                        j = J * 3 + j_local
                        U[i * 6 + j][r] = c_s[I][J]
                        
            # Fill V
            # V[j*6+k, r] = d_s[J][K] if j%3 == j_local
            for J in range(2):
                j = J * 3 + j_local
                for K in range(2):
                    for k_local in range(3):
                        k = K * 3 + k_local
                        V[j * 6 + k][r] = d_s[J][K]
                        
            # Fill W
            # W[k*6+i, r] = a_s[I][K]
            for K in range(2):
                for k_local in range(3):
                    k = K * 3 + k_local
                    for I in range(2):
                        for i_local in range(3):
                            i = I * 3 + i_local
                            W[k * 6 + i][r] = a_s[I][K]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
