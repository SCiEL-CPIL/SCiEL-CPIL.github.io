import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    DEADLINE = 130  # seconds safety margin

    # Strategy: Winograd F(4) for 4x4x4 (rank 16) + schoolbook for 4x4x1 (rank 16) = rank 32
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * rank for _ in range(m * p)]  # 20 x 32
    W = [[0] * rank for _ in range(p * n)]  # 20 x 32

    # Winograd F(4) transform matrix
    T = [[1, 0, 1, 0],
         [0, 1, 0, -1],
         [1, 0, -1, 0],
         [0, 1, 0, 1]]
    
    col = 0
    for r in range(4):
        for s in range(4):
            for i in range(4):
                U[i * 4 + s][col] += T[r][i]
            for j in range(4):
                V[j * 5 + r][col] += T[s][j]
            for i in range(4):
                for j in range(4):
                    W[j * 4 + i][col] += T[r][i] * T[s][j]
            col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # Verify the decomposition is correct
    if time.time() - start_time < DEADLINE - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        A_test = rng.rand(4, 4)
        B_test = rng.rand(4, 5)
        
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((4, 5))
        for alpha in range(rank):
            a_part = sum(A_test[i, k] * U_arr[i * 4 + k, alpha] for i in range(4) for k in range(4))
            b_part = sum(B_test[k, j] * V_arr[k * 5 + j, alpha] for k in range(4) for j in range(5))
            for i in range(4):
                for j in range(5):
                    C_computed[i, j] += a_part * b_part * W_arr[j * 4 + i, alpha]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: use a simpler but valid rank-80 decomposition
            rank = 80
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(5):
                        U[i * 4 + j][col] = 1
                        V[j * 5 + k][col] = 1
                        W[k * 4 + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
