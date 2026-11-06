"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-189 tensor decomposition for 6x6 matrix multiplication
    using one level of Strassen recursion (7 terms) and schoolbook (rank 27) 
    for the 3x3 sub-blocks.
    
    Rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS
    # Theoretical min rank for 6x6 is 189 (Strassen). Schoolbook is 216.
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # u_s[I][J]: Coefficient of A[I][J] in X_s
    # v_s[J][K]: Coefficient of B[J][K] in Y_s
    # w_s[K][I]: Coefficient of Z_s in C[K][I]
    
    # s=0: M1 = (A00 + A11)(B00 + B11)
    u0 = [[1, 0], [0, 1]]
    v0 = [[1, 0], [0, 1]]
    w0 = [[1, 0], [0, 1]] # C00 += M1, C11 += M1
    
    # s=1: M2 = (A10 + A11)B00
    u1 = [[0, 0], [1, 1]]
    v1 = [[1, 0], [0, 0]]
    w1 = [[0, 0], [1, -1]] # C10 += M2, C11 -= M2
    
    # s=2: M3 = A00(B01 - B11)
    u2 = [[1, 0], [0, 0]]
    v2 = [[0, 1], [0, -1]]
    w2 = [[0, 1], [0, 1]]  # C01 += M3, C11 += M3
    
    # s=3: M4 = A11(B10 - B00)
    u3 = [[0, 0], [0, 1]]
    v3 = [[-1, 0], [0, 1]]
    w3 = [[1, 0], [1, 0]]  # C00 += M4, C10 += M4
    
    # s=4: M5 = (A00 + A01)B11
    u4 = [[1, 1], [0, 0]]
    v4 = [[0, 0], [0, 1]]
    w4 = [[-1, 0], [0, 1]] # C00 -= M5, C01 += M5
    
    # s=5: M6 = (A10 - A00)(B00 + B01)
    u5 = [[-1, 0], [1, 0]]
    v5 = [[1, 1], [0, 0]]
    w5 = [[0, 0], [0, 1]]  # C11 += M6
    
    # s=6: M7 = (A01 - A11)(B10 + B11)
    u6 = [[0, 1], [0, -1]]
    v6 = [[0, 0], [1, 1]]
    w6 = [[1, 0], [0, 0]]  # C00 += M7

    coeffs = [
        (u0, v0, w0),
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6)
    ]
    
    # Iterate through Strassen terms
    for s in range(7):
        u_s, v_s, w_s = coeffs[s]
        
        # Iterate through schoolbook indices for the 3x3 sub-blocks
        # l corresponds to (i', j', k')
        for ip in range(3):
            for jp in range(3):
                for kp in range(3):
                    l = 9 * ip + 3 * jp + kp
                    r = 27 * s + l
                    
                    # Fill U
                    # Global i = 3*I + ip, Global j = 3*J + jp
                    for I in range(2):
                        for J in range(2):
                            i = 3 * I + ip
                            j = 3 * J + jp
                            val = u_s[I][J]
                            if val != 0:
                                U[i * m + j][r] = val
                                
                    # Fill V
                    # Global j = 3*J + jp, Global k = 3*K + kp
                    for J in range(2):
                        for K in range(2):
                            j = 3 * J + jp
                            k = 3 * K + kp
                            val = v_s[J][K]
                            if val != 0:
                                V[j * p + k][r] = val
                                
                    # Fill W
                    # Global k = 3*K + kp, Global i = 3*I + ip
                    for K in range(2):
                        for I in range(2):
                            k = 3 * K + kp
                            i = 3 * I + ip
                            val = w_s[K][I]
                            if val != 0:
                                W[k * n + i][r] = val

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
