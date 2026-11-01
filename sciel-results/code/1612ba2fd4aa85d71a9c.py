"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Exact tensor decomposition for 2x5x5 matrix multiplication.
    Returns the schoolbook rank-50 decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    rank_sb = n * m * p  # 50
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Also verify zeros
        for i_idx in range(n*m):
            for j_idx in range(m*p):
                for k_idx in range(p*n):
                    # Check if this corresponds to a valid (i,j,k)
                    i = i_idx // m
                    j = i_idx % m
                    j2 = j_idx // p
                    k = j_idx % p
                    k2 = k_idx // n
                    i2 = k_idx % n
                    if i == j2 and j == k2 and k == i2:
                        continue  # This is a 1-entry, already checked
                    val = 0.0
                    for c in range(r):
                        val += U[i_idx, c] * V[j_idx, c] * W[k_idx, c]
                    if abs(val) > 1e-9:
                        return False
        return True

    if verify(U_sb, V_sb, W_sb):
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
