import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 2x2, grid 3x3
    block_size = 2
    grid_size = 3
    
    rank = grid_size * grid_size * grid_size * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # L_A[t][r, s]
    # L_B[t][s, c]
    # Alpha[t][r, c]
    
    # t=0: M1
    # L_A: A11+A22 -> [[1,0],[1,0]]? No, A11 is (0,0), A22 is (1,1).
    # A = [[A11, A12], [A21, A22]]
    # L_A[0] = [[1, 0], [0, 1]] (A11 + A22)
    # L_B[0] = [[1, 0], [0, 1]] (B11 + B22)
    # Alpha[0]: P11 += 1, P22 += 1. P12, P21 = 0.
    
    # t=1: M2
    # L_A[1] = [[0, 0], [1, 1]] (A21 + A22)
    # L_B[1] = [[1, 0], [0, 0]] (B11)
    # Alpha[1]: P21 += 1.
    
    # t=2: M3
    # L_A[2] = [[1, 0], [0, 0]] (A11)
    # L_B[2] = [[0, 0], [1, -1]] (B21 - B22)
    # Alpha[2]: P12 += 1.
    
    # t=3: M4
    # L_A[3] = [[0, 0], [0, 1]] (A22)
    # L_B[3] = [[-1, 1], [0, 0]] (B12 - B11)
    # Alpha[3]: P11 += 1, P21 += 1.
    
    # t=4: M5
    # L_A[4] = [[1, 1], [0, 0]] (A11 + A12)
    # L_B[4] = [[0, 0], [0, 1]] (B22)
    # Alpha[4]: P11 -= 1, P12 += 1.
    
    # t=5: M6
    # L_A[5] = [[-1, 0], [1, 0]] (A21 - A11)
    # L_B[5] = [[1, 1], [0, 0]] (B11 + B12)
    # Alpha[5]: P22 += 1.
    
    # t=6: M7
    # L_A[6] = [[0, 1], [-1, 0]] (A12 - A22)
    # L_B[6] = [[0, 0], [1, 1]] (B21 + B22)
    # Alpha[6]: P11 += 1, P22 -= 1.
    
    L_A = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [-1, 0]]
    ]
    
    L_B = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [1, -1]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    Alpha = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, 0]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, -1]]
    ]
    
    column = 0
    for a in range(grid_size):
        for k_mid in range(grid_size):
            for b in range(grid_size):
                for t in range(7):
                    # A block indices
                    # Global i = 2*a + r
                    # Global j = 2*k_mid + s
                    for r in range(block_size):
                        for s in range(block_size):
                            gi = 2 * a + r
                            gj = 2 * k_mid + s
                            u_idx = gi * m + gj
                            U[u_idx][column] += L_A[t][r][s]
                    
                    # B block indices
                    # Global j = 2*k_mid + s_prime
                    # Global k = 2*b + c
                    for s_prime in range(block_size):
                        for c in range(block_size):
                            gj = 2 * k_mid + s_prime
                            gk = 2 * b + c
                            v_idx = gj * p + gk
                            V[v_idx][column] += L_B[t][s_prime][c]
                    
                    # C block indices
                    # Global k = 2*b + c
                    # Global i = 2*a + r
                    for r in range(block_size):
                        for c in range(block_size):
                            gk = 2 * b + c
                            gi = 2 * a + r
                            w_idx = gk * n + gi
                            W[w_idx][column] += Alpha[t][r][c]
                    
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
