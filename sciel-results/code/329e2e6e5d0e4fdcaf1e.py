import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 66.
    # Split m=4 into two parts of size 2.
    # Split p=6 into three parts of size 2.
    # For each 3x2x2 block, we use rank 11 (4 for 1x2x2 + 7 for 2x2x2 Strassen).
    # Total rank = 2 (j-parts) * 3 (k-parts) * 11 = 66.
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)] # 12 x 66
    V = [[0] * rank for _ in range(m * p)] # 24 x 66
    W = [[0] * rank for _ in range(p * n)] # 18 x 66
    
    col = 0
    
    # Indices for the 3x2x2 block
    # A part: i in {0, 1, 2}, j in {j0, j1}
    # B part: j in {j0, j1}, k in {k0, k1}
    # C part: i in {0, 1, 2}, k in {k0, k1}
    
    # Strassen coefficients for 2x2x2
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    # x00=A[1,j0], x01=A[1,j1], x10=A[2,j0], x11=A[2,j1]
    # y00=B[j0,k0], y01=B[j0,k1], y10=B[j1,k0], y11=B[j1,k1]
    
    # Z00 -> C[1, k0]
    # Z01 -> C[1, k1]
    # Z10 -> C[2, k0]
    # Z11 -> C[2, k1]
    
    for j_part in range(2):
        j0 = j_part * 2
        j1 = j_part * 2 + 1
        
        for k_part in range(3):
            k0 = k_part * 2
            k1 = k_part * 2 + 1
            
            # --- Part 1: 1x2x2 (i=0) ---
            # 4 terms
            # 1. A[0,j0] * B[j0,k0] -> C[0,k0]
            U[0 * m + j0][col] = 1
            V[j0 * p + k0][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1
            
            # 2. A[0,j1] * B[j1,k0] -> C[0,k0]
            U[0 * m + j1][col] = 1
            V[j1 * p + k0][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1
            
            # 3. A[0,j0] * B[j0,k1] -> C[0,k1]
            U[0 * m + j0][col] = 1
            V[j0 * p + k1][col] = 1
            W[k1 * n + 0][col] = 1
            col += 1
            
            # 4. A[0,j1] * B[j1,k1] -> C[0,k1]
            U[0 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            W[k1 * n + 0][col] = 1
            col += 1
            
            # --- Part 2: 2x2x2 (i=1,2) Strassen ---
            # 7 terms
            
            # m1 = (x00+x11)(y00+y11)
            # x00=A[1,j0], x11=A[2,j1]
            # y00=B[j0,k0], y11=B[j1,k1]
            U[1 * m + j0][col] = 1
            U[2 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            # Z00 (C[1,k0]) += 1
            # Z01 (C[1,k1]) += 1
            # Z10 (C[2,k0]) += -1
            # Z11 (C[2,k1]) += 1
            W[k0 * n + 1][col] = 1
            W[k1 * n + 1][col] = 1
            W[k0 * n + 2][col] = -1
            W[k1 * n + 2][col] = 1
            col += 1
            
            # m2 = (x10+x11)y00
            # x10=A[2,j0], x11=A[2,j1]
            # y00=B[j0,k0]
            U[2 * m + j0][col] = 1
            U[2 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            # Z10 (C[2,k0]) += 1
            # Z11 (C[2,k1]) += -1
            W[k0 * n + 2][col] = 1
            W[k1 * n + 2][col] = -1
            col += 1
            
            # m3 = x00(y01-y11)
            # x00=A[1,j0]
            # y01=B[j0,k1], y11=B[j1,k1]
            U[1 * m + j0][col] = 1
            V[j0 * p + k1][col] = 1
            V[j1 * p + k1][col] = -1
            # Z01 (C[1,k1]) += 1
            # Z11 (C[2,k1]) += 1
            W[k1 * n + 1][col] = 1
            W[k1 * n + 2][col] = 1
            col += 1
            
            # m4 = x11(y10-y00)
            # x11=A[2,j1]
            # y10=B[j1,k0], y00=B[j0,k0]
            U[2 * m + j1][col] = 1
            V[j1 * p + k0][col] = 1
            V[j0 * p + k0][col] = -1
            # Z00 (C[1,k0]) += 1
            # Z10 (C[2,k0]) += 1
            W[k0 * n + 1][col] = 1
            W[k0 * n + 2][col] = 1
            col += 1
            
            # m5 = (x00+x01)y11
            # x00=A[1,j0], x01=A[1,j1]
            # y11=B[j1,k1]
            U[1 * m + j0][col] = 1
            U[1 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            # Z00 (C[1,k0]) += -1
            # Z01 (C[1,k1]) += 1
            W[k0 * n + 1][col] = -1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # m6 = (x10-x00)(y00+y01)
            # x10=A[2,j0], x00=A[1,j0]
            # y00=B[j0,k0], y01=B[j0,k1]
            U[2 * m + j0][col] = 1
            U[1 * m + j0][col] = -1
            V[j0 * p + k0][col] = 1
            V[j0 * p + k1][col] = 1
            # Z11 (C[2,k1]) += 1
            W[k1 * n + 2][col] = 1
            col += 1
            
            # m7 = (x01-x11)(y10+y11)
            # x01=A[1,j1], x11=A[2,j1]
            # y10=B[j1,k0], y11=B[j1,k1]
            U[1 * m + j1][col] = 1
            U[2 * m + j1][col] = -1
            V[j1 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            # Z00 (C[1,k0]) += 1
            W[k0 * n + 1][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
