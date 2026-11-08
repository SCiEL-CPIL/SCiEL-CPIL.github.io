"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # A: 2x3, B: 3x6, C: 2x6
    # U: 6 x rank, V: 18 x rank, W: 12 x rank
    # W convention: W[k*n+i, r]
    
    # Rank-18 decomposition: 3 blocks of naive 2x3 x 3x2 -> 2x2 (rank 6 each)
    rank = 18
    U = [[0] * rank for _ in range(n * m)]   # 6 x 18
    V = [[0] * rank for _ in range(m * p)]   # 18 x 18
    W = [[0] * rank for _ in range(p * n)]   # 12 x 18
    
    for block in range(3):
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    r = block * 6 + i * m * 2 + j * 2 + k
                    U[i * m + j][r] = 1
                    # B global index: j * p + (block * 2 + k)
                    V[j * p + (block * 2 + k)][r] = 1
                    # C global index: (block * 2 + k) * n + i
                    W[(block * 2 + k) * n + i][r] = 1
    
    # Verify the decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_val = np.dot(U_arr[:, r], A_flat)
        v_val = np.dot(V_arr[:, r], B_flat)
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += W_arr[k * n + i, r] * u_val * v_val
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to naive rank 36
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_naive[i * m + j][r] = 1
                    V_naive[j * p + k][r] = 1
                    W_naive[k * n + i][r] = 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

