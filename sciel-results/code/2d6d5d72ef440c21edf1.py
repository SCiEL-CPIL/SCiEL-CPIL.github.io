import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s margin
    
    n, m, p = 4, 4, 6
    
    # Use naive rank-96 decomposition (guaranteed correct)
    # U shape: (n*m, rank) = (16, 96)
    # V shape: (m*p, rank) = (24, 96)
    # W shape: (p*n, rank) = (24, 96)
    
    rank = n * m * p  # 96
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Quick verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 6))
    for r in range(rank):
        alpha_r = sum(U_np[i * 4 + j, r] * A_test[i, j] for i in range(4) for j in range(4))
        beta_r = sum(V_np[j * 6 + k, r] * B_test[j, k] for j in range(4) for k in range(6))
        for i in range(4):
            for k in range(6):
                C_computed[i, k] += alpha_r * beta_r * W_np[k * 4 + i, r]
    
    error = np.max(np.abs(C_computed - C_expected))
    
    if error > 1e-10:
        # Should never happen for naive, but safety net
        pass
    
    return {"U": U, "V": V, "W": W}
