"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The tensor decomposes into 5 independent 2x2x2 matrix multiplication blocks.
    Each block uses Strassen's rank-7 algorithm, giving total rank 5*7 = 35.
    This is the proven minimum rank for this tensor.
    
    Interface:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Index conventions:
    U[i*m + j, r] for A[i,j]
    V[j*p + k, r] for B[j,k]
    W[k*n + i, r] for C[i,k]  (official trace convention)
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of 2 columns each
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    # Strassen's 7 products for M1 * M2 where:
    # M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # Result:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U coefficients: [a, b, c, d] for each Strassen product
    strassen_U = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1]   # P7: b-d
    ]
    
    # V coefficients: [e, f, g, h] for each Strassen product
    strassen_V = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1]    # P7: g+h
    ]
    
    # W coefficients: contribution to [C00, C01, C10, C11]
    strassen_W = [
        [1, 0, 0, 1],   # P1: C00+1, C11+1
        [0, 0, 1, -1],  # P2: C10+1, C11-1
        [0, 1, 0, 1],   # P3: C01+1, C11+1
        [1, 0, 1, 0],   # P4: C00+1, C10+1
        [-1, 1, 0, 0],  # P5: C00-1, C01+1
        [0, 0, 0, 1],   # P6: C11+1
        [1, 0, 0, 0]    # P7: C00+1
    ]
    
    for b in range(num_blocks):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U: index i*m + j (i in 0..1, j in 0..1)
            U[0][col] = strassen_U[r][0]  # A[0,0] = a
            U[1][col] = strassen_U[r][1]  # A[0,1] = b
            U[2][col] = strassen_U[r][2]  # A[1,0] = c
            U[3][col] = strassen_U[r][3]  # A[1,1] = d
            
            # Fill V: index j*p + k (j in 0..1, k in 0..9)
            k0 = 2 * b      # local column 0 -> global column 2*b
            k1 = 2 * b + 1  # local column 1 -> global column 2*b+1
            V[0 * p + k0][col] = strassen_V[r][0]  # B[0, k0] = e
            V[0 * p + k1][col] = strassen_V[r][1]  # B[0, k1] = f
            V[1 * p + k0][col] = strassen_V[r][2]  # B[1, k0] = g
            V[1 * p + k1][col] = strassen_V[r][3]  # B[1, k1] = h
            
            # Fill W: index k*n + i (k in 0..9, i in 0..1)
            # C[0, k0]: i=0, k=k0 -> idx k0*n + 0
            # C[0, k1]: i=0, k=k1 -> idx k1*n + 0
            # C[1, k0]: i=1, k=k0 -> idx k0*n + 1
            # C[1, k1]: i=1, k=k1 -> idx k1*n + 1
            W[k0 * n + 0][col] = strassen_W[r][0]  # C[0, k0]
            W[k1 * n + 0][col] = strassen_W[r][1]  # C[0, k1]
            W[k0 * n + 1][col] = strassen_W[r][2]  # C[1, k0]
            W[k1 * n + 1][col] = strassen_W[r][3]  # C[1, k1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
