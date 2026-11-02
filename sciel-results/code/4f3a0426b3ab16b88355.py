"""Tensor decomposition for 2x3x8 matrix multiplication using a rank-44 construction."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    
    # Part 1: A_L (2x2) * B_T (2x8)
    # A_L indices: i in 0..1, j in 0..1
    # B_T indices: j in 0..1, k in 0..7
    # Split k into 4 groups of 2
    for g in range(4):
        k1 = 2 * g
        k2 = 2 * g + 1
        
        # Strassen's 7 terms for 2x2x2
        # a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
        # b11=B[0,k1], b12=B[0,k2], b21=B[1,k1], b22=B[1,k2]
        
        # m1 = a11 * (b12 - b22)
        U[0*3+0][col] += 1
        V[0*8+k2][col] += 1
        V[1*8+k2][col] += -1
        # C contributions: C[0,k2] += m1, C[1,k2] += m1
        W[k2*2+0][col] += 1
        W[k2*2+1][col] += 1
        col += 1

        # m2 = (a11 + a12) * b22
        U[0*3+0][col] += 1
        U[0*3+1][col] += 1
        V[1*8+k2][col] += 1
        # C contributions: C[0,k1] -= m2, C[0,k2] += m2
        W[k1*2+0][col] += -1
        W[k2*2+0][col] += 1
        col += 1

        # m3 = (a21 + a22) * b11
        U[1*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*8+k1][col] += 1
        # C contributions: C[1,k1] += m3, C[1,k2] -= m3
        W[k1*2+1][col] += 1
        W[k2*2+1][col] += -1
        col += 1

        # m4 = a22 * (b21 - b11)
        U[1*3+1][col] += 1
        V[1*8+k1][col] += 1
        V[0*8+k1][col] += -1
        # C contributions: C[0,k1] += m4, C[1,k1] += m4
        W[k1*2+0][col] += 1
        W[k1*2+1][col] += 1
        col += 1

        # m5 = (a11 + a22) * (b11 + b22)
        U[0*3+0][col] += 1
        U[1*3+1][col] += 1
        V[0*8+k1][col] += 1
        V[1*8+k2][col] += 1
        # C contributions: C[0,k1] += m5, C[0,k2] -= m5, C[1,k1] += m5, C[1,k2] -= m5
        W[k1*2+0][col] += 1
        W[k2*2+0][col] += -1
        W[k1*2+1][col] += 1
        W[k2*2+1][col] += -1
        col += 1

        # m6 = (a12 - a22) * (b21 + b22)
        U[0*3+1][col] += 1
        U[1*3+1][col] += -1
        V[1*8+k1][col] += 1
        V[1*8+k2][col] += 1
        # C contributions: C[0,k1] += m6
        W[k1*2+0][col] += 1
        col += 1

        # m7 = (a21 - a11) * (b11 + b12)
        U[1*3+0][col] += 1
        U[0*3+0][col] += -1
        V[0*8+k1][col] += 1
        V[0*8+k2][col] += 1
        # C contributions: C[1,k2] += m7
        W[k2*2+1][col] += 1
        col += 1

    # Part 2: A_R (2x1) * B_B (1x8)
    # A_R index: i in 0..1, j=2
    # B_B index: j=2, k in 0..7
    # C[i,k] = A[i,2] * B[2,k]
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*2+i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
