import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Rank 1 for j=0
    # U[i*m + 0, 0] = 1
    # V[0*p + k, 0] = 1
    # W[k*n + i, 0] = 1
    for i in range(n):
        U[i * m + 0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
        for i in range(n):
            W[k * n + i][0] = 1

    # Part 2 & 3: Strassen for j in {1, 2}
    # Strassen definitions for 2x2 matrices
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x21+x22)y11
    # M3 = x11(y12-y22)
    # M4 = x22(y21-y11)
    # M5 = (x11+x12)y22
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12-x22)(y21+y22)
    
    # W coefficients (2x2) for each M
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # w[r][k][i] where r is 0..6, k is 0..1 (local), i is 0..1 (local)
    # Global i is 0..1. Local i is same.
    # Local k maps to global k.
    
    # Coefficients for W (local k, local i)
    # r=0 (M1): Z11+=1, Z22+=1 -> w[0][0]=1, w[0][1]=0 (for i=0,1? No, i is row of Z)
    # Z is 2x2. Z[0][0], Z[0][1], Z[1][0], Z[1][1].
    # w[r][k][i] is coeff of Z[i][k].
    
    w_coeffs = [
        # M1
        [[1, 0], [0, 1]], # k=0: i=0->1, i=1->0? No.
        # Z11 (i=0, k=0) += 1. Z22 (i=1, k=1) += 1.
        # So w[0][0][0]=1, w[0][1][1]=1. Others 0.
        # Let's use a 2x2 matrix for each r.
    ]
    
    # Define W contributions as 2x2 matrices for each r
    W_local = [
        [[1, 0], [0, 1]], # M1: Z11, Z22
        [[0, 0], [1, 0]], # M2: Z21
        [[0, 1], [0, 1]], # M3: Z12, Z22
        [[1, 0], [1, 0]], # M4: Z11, Z21
        [[-1, 0], [0, 1]],# M5: Z11(-), Z12(+) -> Wait. Z11 = ... - M5. Z12 = ... + M5.
        # So M5 contributes -1 to Z11, +1 to Z12.
        # Z11 is i=0, k=0. Z12 is i=0, k=1.
        # So w[4][0][0] = -1, w[4][1][0] = 1.
        # My matrix above [[-1, 0], [0, 1]] is w[k][i]?
        # Let's stick to w[r][i][k] or w[r][k][i].
        # Let's use w[r][i][k].
    ]
    
    # Let's redefine W_local[r][i][k]
    W_local = [
        # M1
        [[1, 0], [0, 1]], # i=0: k=0->1, k=1->0. i=1: k=0->0, k=1->1.
        # Wait. Z11 (i=0,k=0) += 1. Z22 (i=1,k=1) += 1.
        # So W[0][0][0]=1, W[0][1][1]=1.
        # Matrix: [[1,0],[0,1]] if row is i, col is k.
        
        # M2
        [[0, 0], [1, 0]], # Z21 (i=1,k=0) += 1.
        
        # M3
        [[0, 1], [0, 1]], # Z12 (i=0,k=1) += 1. Z22 (i=1,k=1) += 1.
        
        # M4
        [[1, 0], [1, 0]], # Z11 (i=0,k=0) += 1. Z21 (i=1,k=0) += 1.
        
        # M5
        [[-1, 1], [0, 0]],# Z11 (i=0,k=0) -= 1. Z12 (i=0,k=1) += 1.
        
        # M6
        [[0, 0], [0, 1]], # Z22 (i=1,k=1) += 1.
        
        # M7
        [[1, 0], [0, 0]], # Z11 (i=0,k=0) += 1.
    ]
    
    # U coefficients (2x2) for each r
    # M1: (x11+x22) -> u11=1, u22=1.
    # M2: (x21+x22) -> u21=1, u22=1.
    # M3: x11 -> u11=1.
    # M4: x22 -> u22=1.
    # M5: (x11+x12) -> u11=1, u12=1.
    # M6: (x21-x11) -> u21=1, u11=-1.
    # M7: (x12-x22) -> u12=1, u22=-1.
    
    U_local = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    
    # V coefficients (2x2) for each r
    # M1: (y11+y22) -> v11=1, v22=1.
    # M2: y11 -> v11=1.
    # M3: (y12-y22) -> v12=1, v22=-1.
    # M4: (y21-y11) -> v21=1, v11=-1.
    # M5: y22 -> v22=1.
    # M6: (y11+y12) -> v11=1, v12=1.
    # M7: (y21+y22) -> v21=1, v22=1.
    
    V_local = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[-1, 0], [1, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    
    def add_strassen_block(start_col, j_offset, k_offset):
        # j_offset is global j start (1 or 2)
        # k_offset is global k start (0 or 2)
        for r in range(7):
            col = start_col + r
            # Update U
            for i in range(2):
                for j in range(2):
                    val = U_local[r][i][j]
                    if val != 0:
                        U[i * m + (j_offset + j)][col] += val
            # Update V
            for j in range(2):
                for k in range(2):
                    val = V_local[r][j][k]
                    if val != 0:
                        V[(j_offset + j) * p + (k_offset + k)][col] += val
            # Update W
            for i in range(2):
                for k in range(2):
                    val = W_local[r][i][k]
                    if val != 0:
                        W[(k_offset + k) * n + i][col] += val

    # Add block for k in {0, 1}
    add_strassen_block(1, 1, 0)
    # Add block for k in {2, 3}
    add_strassen_block(8, 1, 2)

    return {"U": U, "V": V, "W": W}
