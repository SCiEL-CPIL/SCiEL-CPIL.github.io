import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The tensor decomposes into 5 independent 2x2x2 multiplication tensors.
    Since the rank of 2x2x2 is 7 (Strassen), the total rank is 5*7=35.
    
    U shape: (4, 35)
    V shape: (20, 35)
    W shape: (20, 35)
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 7 products
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    
    strassen_A = [
        [1, 0, 0, 1],   # P1: (a+d)
        [0, 0, 1, 1],   # P2: (c+d)
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: (a+b)
        [-1, 0, 1, 0],  # P6: (c-a)
        [0, 1, 0, -1]   # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P1: (e+h)
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: (f-h)
        [-1, 0, 1, 0],  # P4: (g-e)
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: (e+f)
        [0, 0, 1, 1]    # P7: (g+h)
    ]
    
    # W contributions: w_contrib[r] = [C00, C01, C10, C11]
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    w_contrib = [
        [1,  0,  0,  1],  # P1
        [0,  0,  1, -1],  # P2
        [0,  1,  0,  1],  # P3
        [1,  0,  1,  0],  # P4
        [-1, 1,  0,  0],  # P5
        [0,  0,  0,  1],  # P6
        [1,  0,  0,  0]   # P7
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U: index i*m + j
            U[0][col] = strassen_A[r][0]  # a
            U[1][col] = strassen_A[r][1]  # b
            U[2][col] = strassen_A[r][2]  # c
            U[3][col] = strassen_A[r][3]  # d
            
            # Fill V: index j*p + k
            k0 = 2 * b
            k1 = 2 * b + 1
            
            V[0 * p + k0][col] = strassen_B[r][0]  # e
            V[0 * p + k1][col] = strassen_B[r][1]  # f
            V[1 * p + k0][col] = strassen_B[r][2]  # g
            V[1 * p + k1][col] = strassen_B[r][3]  # h
            
            # Fill W: index k*n + i (official trace convention)
            # C00: k_local=0, i=0 -> k_global=2*b -> idx (2*b)*2 + 0
            # C01: k_local=1, i=0 -> k_global=2*b+1 -> idx (2*b+1)*2 + 0
            # C10: k_local=0, i=1 -> k_global=2*b -> idx (2*b)*2 + 1
            # C11: k_local=1, i=1 -> k_global=2*b+1 -> idx (2*b+1)*2 + 1
            
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]    # C00
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]  # C01
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]    # C10
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]  # C11
    
    return {"U": U, "V": V, "W": W}
