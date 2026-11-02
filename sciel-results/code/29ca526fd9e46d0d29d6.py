"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    def verify_decomp(U, V, W, n, m, p):
        """Check if the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for c in range(U.shape[1]):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Build rank-28 Strassen decomposition as baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))

    def strassen_2x2_decomp(P_rows, P_cols, Q_rows, Q_cols, R_rows, R_cols, base_col, U, V, W):
        u11 = P_rows[0] * m + P_cols[0]
        u12 = P_rows[0] * m + P_cols[1]
        u21 = P_rows[1] * m + P_cols[0]
        u22 = P_rows[1] * m + P_cols[1]
        v11 = Q_rows[0] * p + Q_cols[0]
        v12 = Q_rows[0] * p + Q_cols[1]
        v21 = Q_rows[1] * p + Q_cols[0]
        v22 = Q_rows[1] * p + Q_cols[1]
        w11 = R_cols[0] * n + R_rows[0]
        w12 = R_cols[1] * n + R_rows[0]
        w21 = R_cols[0] * n + R_rows[1]
        w22 = R_cols[1] * n + R_rows[1]

        c = base_col + 0
        U[u11, c] += 1; U[u22, c] += 1; V[v11, c] += 1; V[v22, c] += 1; W[w11, c] += 1; W[w22, c] += 1
        c = base_col + 1
        U[u21, c] += 1; U[u22, c] += 1; V[v11, c] += 1; W[w21, c] += 1; W[w22, c] += -1
        c = base_col + 2
        U[u11, c] += 1; V[v12, c] += 1; V[v22, c] += -1; W[w12, c] += 1; W[w22, c] += 1
        c = base_col + 3
        U[u22, c] += 1; V[v21, c] += 1; V[v11, c] += -1; W[w11, c] += 1; W[w21, c] += 1
        c = base_col + 4
        U[u11, c] += 1; U[u12, c] += 1; V[v22, c] += 1; W[w11, c] += -1; W[w12, c] += 1
        c = base_col + 5
        U[u21, c] += 1; U[u11, c] += -1; V[v11, c] += 1; V[v12, c] += 1; W[w22, c] += 1
        c = base_col + 6
        U[u12, c] += 1; U[u22, c] += -1; V[v21, c] += 1; V[v22, c] += 1; W[w11, c] += 1; W[w22, c] += -1
        return base_col + 7

    col = strassen_2x2_decomp([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0, U28, V28, W28)
    col = strassen_2x2_decomp([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], col, U28, V28, W28)
    col = strassen_2x2_decomp([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], col, U28, V28, W28)
    col = strassen_2x2_decomp([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], col, U28, V28, W28)

    if not verify_decomp(U28, V28, W28, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END
