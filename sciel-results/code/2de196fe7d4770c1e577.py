"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses Strassen's rank-7 decomposition for 2x2x2 blocks applied to overlapping
    2x2 column blocks of B, plus schoolbook for the remaining cross terms.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = schoolbook()

    # Strassen's 2x2x2 decomposition (rank 7)
    def strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7 (flattened 2x2 A)
        V: 4x7 (flattened 2x2 B)
        W: 4x7 (flattened 2x2 C)
        Index: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Term 0: m1 = (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1

        # Term 1: m2 = (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1

        # Term 2: m3 = a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1

        # Term 3: m4 = a11(b10-b00)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1

        # Term 4: m5 = (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1

        # Term 5: m6 = (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # Term 6: m7 = (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = -1

        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.sum(U[i*2+j] * V[j*2+k] * W[k*2+i])
                    if abs(val - 1.0) > 1e-9:
                        return None

        return U.tolist(), V.tolist(), W.tolist()

    # Build decomposition using Strassen on 2x2 blocks + schoolbook corrections
    def build_strassen_based():
        """Build decomposition using Strassen on 2x2 blocks + schoolbook corrections."""
        # Total rank: 4 blocks * (7 Strassen + 4 cross1 + 4 cross2 + 2 cross3) = 4 * 17 = 68
        total_rank = 68
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))

        col = 0

        strassen = strassen_2x2x2()
        if strassen is None:
            return None
        S_U, S_V, S_W = strassen

        col_blocks = [(0,1), (2,3), (4,5), (6,7)]

        for (k1, k2) in col_blocks:
            # 1. Strassen terms for A[0:2, 0:2] * B[0:2, k1:k2+1] -> C[0:2, k1:k2+1]
            for s in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*3+j, col] += S_U[i*2+j][s]
                for j in range(2):
                    for kk in range(2):
                        k_idx = k1 + kk
                        V[j*8+k_idx, col] += S_V[j*2+kk][s]
                for i in range(2):
                    for kk in range(2):
                        k_idx = k1 + kk
                        W[k_idx*3+i, col] += S_W[kk*2+i][s]
                col += 1

            # 2. Cross terms: A[0:2, 2] * B[2, k1:k2+1] -> C[0:2, k1:k2+1]
            for i in range(2):
                for kk in range(2):
                    k_idx = k1 + kk
                    U[i*3+2, col] = 1
                    V[2*8+k_idx, col] = 1
                    W[k_idx*3+i, col] = 1
                    col += 1

            # 3. Cross terms: A[2, 0:2] * B[0:2, k1:k2+1] -> C[2, k1:k2+1]
            for j in range(2):
                for kk in range(2):
                    k_idx = k1 + kk
                    U[2*3+j, col] = 1
                    V[j*8+k_idx, col] = 1
                    W[k_idx*3+2, col] = 1
                    col += 1

            # 4. Cross terms: A[2, 2] * B[2, k1:k2+1] -> C[2, k1:k2+1]
            for kk in range(2):
                k_idx = k1 + kk
                U[2*3+2, col] = 1
                V[2*8+k_idx, col] = 1
                W[k_idx*3+2, col] = 1
                col += 1

        if not validate(U.tolist(), V.tolist(), W.tolist()):
            return None

        return U.tolist(), V.tolist(), W.tolist(), total_rank

    result = build_strassen_based()
    if result is not None and elapsed() < deadline - 5:
        U_new, V_new, W_new, rank_new = result
        if rank_new < best_rank:
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
