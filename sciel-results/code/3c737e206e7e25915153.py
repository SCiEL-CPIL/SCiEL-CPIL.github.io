import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block-Strassen decomposition for 6x6 matrices
    # Partition into 3x3 grid of 2x2 blocks
    # Rank = 3*3*3 (block multiplications) * 7 (Strassen rank) = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # u_s, v_s, w_s are vectors of length 4 corresponding to block elements [00, 01, 10, 11]
    # Index mapping: 0->00, 1->01, 2->10, 3->11
    
    u_coeffs = [
        [1, 0, 0, 1],  # M1: a00 + a11
        [0, 0, 1, 1],  # M2: a10 + a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00 + a01
        [-1, 0, 1, 0], # M6: a10 - a00
        [0, 1, 0, -1]  # M7: a01 - a11
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],  # M1: b00 + b11
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: b01 - b11
        [-1, 0, 1, 0], # M4: b10 - b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00 + b01
        [0, 0, 1, 1]   # M7: b10 + b11
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],  # M1: c00 + c11
        [0, 0, 1, -1], # M2: c10 - c11
        [0, 1, 0, 1],  # M3: c01 + c11
        [1, 0, 1, 0],  # M4: c00 + c10
        [-1, 1, 0, 0], # M5: -c00 + c01
        [0, 0, 0, 1],  # M6: c11
        [1, 0, 0, 0]   # M7: c00
    ]
    
    # Iterate over block indices I, J, K and Strassen step s
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    r = ((I * 3 + J) * 3 + K) * 7 + s
                    
                    u_vec = u_coeffs[s]
                    v_vec = v_coeffs[s]
                    w_vec = w_coeffs[s]
                    
                    # Update U
                    for u in range(2):
                        for v in range(2):
                            i = 2 * I + u
                            j = 2 * J + v
                            U[i * m + j][r] = u_vec[2 * u + v]
                            
                    # Update V
                    for v in range(2):
                        for w in range(2):
                            j = 2 * J + v
                            k = 2 * K + w
                            V[j * p + k][r] = v_vec[2 * v + w]
                            
                    # Update W
                    for w in range(2):
                        for u in range(2):
                            k = 2 * K + w
                            i = 2 * I + u
                            W[k * n + i][r] = w_vec[2 * w + u]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
