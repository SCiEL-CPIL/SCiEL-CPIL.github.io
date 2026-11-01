import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # S: 7 x 2 x 2 (for A)
    # T: 7 x 2 x 2 (for B)
    # R: 7 x 2 x 2 (for C)
    
    S = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [1, -1]], # M6
        [[0, 1], [-1, 0]]  # M7
    ]
    
    T = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[0, 1], [0, 1]],  # M6
        [[1, 0], [1, 0]]   # M7
    ]
    
    R = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, -1]], # M2
        [[0, 1], [0, 1]],  # M3
        [[1, 0], [1, 0]],  # M4
        [[-1, 1], [0, 0]], # M5
        [[0, 0], [0, 1]],  # M6
        [[1, 0], [0, 0]]   # M7
    ]
    
    for l in range(p // 2):
        for r in range(7):
            col = l * 7 + r
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = S[r][i][j]
            
            # Fill V
            # V index: j' * p + k
            # k is 2l or 2l+1
            for jp in range(2):
                for k_rel in range(2):
                    k = 2 * l + k_rel
                    V[jp * p + k][col] = T[r][jp][k_rel]
            
            # Fill W
            # W index: k' * n + i'
            # k' is 2l or 2l+1
            for kp in range(2):
                for i_rel in range(2):
                    k_prime = 2 * l + kp
                    W[k_prime * n + i_rel][col] = R[r][i_rel][kp]
    
    return {"U": U, "V": V, "W": W}
