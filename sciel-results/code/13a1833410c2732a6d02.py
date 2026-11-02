import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    # Rank = 27 block multiplications * 7 Strassen terms = 189
    rank = 189
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U, V, W coefficients for each of the 7 terms
    # Local indices (0,0), (0,1), (1,0), (1,1)
    # Flattened local index: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Term 0: M0 = (a00 + a11)(b00 + b11)
    # U: 1 at (0,0), (1,1) -> indices 0, 3
    # V: 1 at (0,0), (1,1) -> indices 0, 3
    # W: 1 at (0,0), (1,1) -> indices 0, 3
    strassen_U = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 1, 0, -1] # M6
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M0
        [1, 0, 0, 0], # M1
        [0, 0, 1, -1],# M2
        [-1, 1, 0, 0],# M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, 1]  # M6
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # M0 -> C00, C11
        [0, 0, 1, -1],# M1 -> C10, C11
        [0, 1, 0, 1], # M2 -> C01, C11
        [1, 0, 1, 0], # M3 -> C00, C10
        [-1, 1, 0, 0],# M4 -> C00, C01
        [0, 0, 0, 1], # M5 -> C11
        [1, 0, 0, 0]  # M6 -> C00
    ]
    
    r = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    # U
                    for lr in range(2):
                        for lc in range(2):
                            val = strassen_U[s][lr*2 + lc]
                            if val != 0:
                                I = 2 * i + lr
                                J = 2 * k + lc
                                U[I * m + J][r] += val
                                
                    # V
                    for lr in range(2):
                        for lc in range(2):
                            val = strassen_V[s][lr*2 + lc]
                            if val != 0:
                                K = 2 * k + lr
                                L = 2 * j + lc
                                V[K * p + L][r] += val
                                
                    # W
                    for lr in range(2):
                        for lc in range(2):
                            val = strassen_W[s][lr*2 + lc]
                            if val != 0:
                                M = 2 * i + lr
                                N = 2 * j + lc
                                # W index is k*n + i in seed, which is col_B * n + row_A
                                # Here N is col of C (col of B), M is row of C (row of A)
                                W[N * n + M][r] += val
                                
                    r += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
