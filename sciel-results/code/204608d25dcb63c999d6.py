import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 1: Handle j=1 (first col of A, first row of B)
    # U[i, 1, 0] = 1
    # V[1, k, 0] = 1
    # W[k, i, 0] = 1
    for i in range(n):
        U[i * m + 1][0] = 1
    for k in range(p):
        V[1 * p + k][0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
            
    # Strassen for j in {2, 3}
    # Split B into 5 blocks of 2 columns each
    num_blocks = p // 2
    block_offset = 0
    r = 1
    for s in range(num_blocks):
        for m_strassen in range(7):
            # Define U, V, W for this term
            # m_strassen 0..6 corresponds to P1..P7
            
            # Coefficients for A (local indices i', j' -> global i, j=j'+1)
            # A_2 is 2x2.
            # P1: (a11+a22)(b11+b22)
            # P2: (a21+a22)b11
            # P3: a11(b12-b22)
            # P4: a22(b21-b11)
            # P5: (a11+a12)b22
            # P6: (a21-a11)(b11+b12)
            # P7: (a12-a22)(b21+b22)
            
            # Map local A indices (0,0)->(1,2), (0,1)->(1,3), (1,0)->(2,2), (1,1)->(2,3)
            # Global U index: i*m + j. i in 0..1, j in 1..2 (local 0..1)
            
            # U coefficients
            if m_strassen == 0: # P1
                U[0*m + 2][r] = 1 # a11
                U[1*m + 3][r] = 1 # a22
            elif m_strassen == 1: # P2
                U[1*m + 2][r] = 1 # a21
                U[1*m + 3][r] = 1 # a22
            elif m_strassen == 2: # P3
                U[0*m + 2][r] = 1 # a11
            elif m_strassen == 3: # P4
                U[1*m + 3][r] = 1 # a22
            elif m_strassen == 4: # P5
                U[0*m + 2][r] = 1 # a11
                U[0*m + 3][r] = 1 # a12
            elif m_strassen == 5: # P6
                U[1*m + 2][r] = 1 # a21
                U[0*m + 2][r] = -1 # -a11
            elif m_strassen == 6: # P7
                U[0*m + 3][r] = 1 # a12
                U[1*m + 3][r] = -1 # -a22

            # V coefficients
            # Local B indices: b11->(2, k), b12->(2, k+1), b21->(3, k), b22->(3, k+1)
            # Global V index: j*p + k. j in 1..2 (local 0..1), k in block
            k0 = s * 2
            k1 = s * 2 + 1
            
            if m_strassen == 0: # P1
                V[2*p + k0][r] = 1 # b11
                V[3*p + k1][r] = 1 # b22
            elif m_strassen == 1: # P2
                V[2*p + k0][r] = 1 # b11
            elif m_strassen == 2: # P3
                V[2*p + k1][r] = 1 # b12
                V[3*p + k1][r] = -1 # -b22
            elif m_strassen == 3: # P4
                V[3*p + k0][r] = 1 # b21
                V[2*p + k0][r] = -1 # -b11
            elif m_strassen == 4: # P5
                V[3*p + k1][r] = 1 # b22
            elif m_strassen == 5: # P6
                V[2*p + k0][r] = 1 # b11
                V[2*p + k1][r] = 1 # b12
            elif m_strassen == 6: # P7
                V[3*p + k0][r] = 1 # b21
                V[3*p + k1][r] = 1 # b22

            # W coefficients
            # C_11 = P1 - P4 + P6 + P7
            # C_12 = P3 + P5 - P7
            # C_21 = P2 + P4 + P6
            # C_22 = P1 - P3 + P2 + P5
            # Global W index: k*n + i. k in block, i in 0..1
            
            if m_strassen == 0: # P1
                W[k0*n + 0][r] = 1  # C11
                W[k1*n + 0][r] = 0  # C12
                W[k0*n + 1][r] = 0  # C21
                W[k1*n + 1][r] = 1  # C22
            elif m_strassen == 1: # P2
                W[k0*n + 1][r] = 1  # C21
                W[k1*n + 1][r] = 1  # C22
            elif m_strassen == 2: # P3
                W[k1*n + 0][r] = 1  # C12
                W[k1*n + 1][r] = -1 # C22
            elif m_strassen == 3: # P4
                W[k0*n + 0][r] = -1 # C11
                W[k0*n + 1][r] = 1  # C21
            elif m_strassen == 4: # P5
                W[k1*n + 0][r] = 1  # C12
                W[k1*n + 1][r] = 1  # C22
            elif m_strassen == 5: # P6
                W[k0*n + 0][r] = 1  # C11
                W[k0*n + 1][r] = 1  # C21
            elif m_strassen == 6: # P7
                W[k0*n + 0][r] = 1  # C11
                W[k1*n + 0][r] = -1 # C12

            r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
