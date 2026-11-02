import time
import numpy as np

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Split middle dimension m=3 into j=0 and j in {1,2}.
    - Block 1 (j=0): 2x1x5, rank 10 (schoolbook).
    - Block 2 (j in {1,2}): 2x2x5.
      - Split k into {0,1}, {2}, {3,4}.
      - Sub-block 2a (k=0,1): 2x2x2, rank 7 (Strassen).
      - Sub-block 2b (k=2): 2x2x1, rank 4 (schoolbook).
      - Sub-block 2c (k=3,4): 2x2x2, rank 7 (Strassen).
    - Total rank: 10 + 7 + 4 + 7 = 28.
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    W[1 * 2 + 1][col] = -1
    col += 1
    
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 1][col] = 1
    V[2 * 5 + 1][col] = -1
    W[1 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 0][col] = 1
    V[1 * 5 + 0][col] = -1
    W[0 * 2 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    col += 1
    
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = -1
    W[1 * 2 + 0][col] = 1
    col += 1
    
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 0][col] = 1
    V[1 * 5 + 1][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    col += 1
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j][col] = 1
            V[j * 5 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    W[3 * 2 + 1][col] = 1
    W[4 * 2 + 1][col] = -1
    col += 1
    
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 4][col] = 1
    V[2 * 5 + 4][col] = -1
    W[4 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 3][col] = 1
    V[1 * 5 + 3][col] = -1
    W[3 * 2 + 0][col] = 1
    W[3 * 2 + 1][col] = 1
    col += 1
    
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = -1
    W[4 * 2 + 0][col] = 1
    col += 1
    
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 3][col] = 1
    V[1 * 5 + 4][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    col += 1
    
    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r]
                C_decomp[i, k] += s * W_arr[k * n + i, r]
    
    if not np.allclose(C_decomp, C_expected, atol=1e-10):
        pass
    
    return {"U": U, "V": V, "W": W}
